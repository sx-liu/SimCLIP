import numpy
import random

import os
import json
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from models.utils import calculate_similarity_matrix, find_top_k_nns

from transformers import AutoFeatureExtractor, AutoTokenizer
from collections import defaultdict


class CLIPDataset(Dataset):
    def __init__(self, input_filename, base_image_dir, img_key,
                 caption_keys, clip_model: str, max_len: int = 77,
                 sep="\t", precision: str = 'fp32', transform=None,
                 random_caption=False):
        # Load data frame from csv/tsv file
        print(f'Loading tsv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.base_image_dir = base_image_dir
        self.images = df[img_key].tolist()
        if not isinstance(caption_keys, list):
            caption_keys = [caption_keys]
        self.captions = df[caption_keys].values.tolist()
        assert len(self.images) == len(self.captions)

        # Load huggingface models
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)

        self.max_len = max_len
        self.precision = precision
        self.transform = transform

        if self.transform is not None:
            print('Applying image transform.')

        self.random_caption = random_caption
        if self.random_caption:
            print(f'Randomly picking captions from {caption_keys}.')

        self.cache = df
        self.original_length = len(self.captions)
        print('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while True:
            if idx < 0:
                idx += len(self)

            if idx < self.original_length:
                image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
            else:
                image_path =str(self.images[idx])

            caption = random.sample(self.captions[idx], 1)[0] if self.random_caption else self.captions[idx][0]
            caption = str(caption)

            try:
                img = Image.open(image_path)

                if self.transform is not None:
                    img = self.transform(img)

                images = self.feature_extractor(
                    img.convert('RGB'),
                    return_tensors="pt").pixel_values[0, ...]

                tokenized_data = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len)
                tokens = tokenized_data.input_ids[0]

                caption_len = tokenized_data.attention_mask[0].sum()

                return image_path, images, tokens, caption_len
            except Exception as e:
                print(f'Error reading {image_path} with caption {caption}: {e}')
                # Pick a new example at random.
                idx = np.random.randint(0, len(self) - 1)

    def inject_mnist(self, num_data: int):
        # Download MNIST data
        mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)

        digit_count = defaultdict(int)

        # Inject MNIST data into the dataset
        for i in range(min(num_data, len(mnist_dataset))):
            img, label = mnist_dataset[i]
            image_path = f'mnist_data/MNIST_{i}.png'
            caption = f"A MNIST digit of {label}"

            # Append the new data to the dataset's internal lists
            self.images.append(image_path)
            self.captions.append([caption])

            img.save(image_path)
            digit_count[label] += 1

        # Print the summary of digits injected
        print(f'Injected {min(num_data, len(mnist_dataset))} MNIST samples into the dataset.')
        for digit, count in sorted(digit_count.items()):
            print(f'Digit {digit}: {count} samples')

        return digit_count


class CLIPEmbeddingDataset(CLIPDataset):
    def __init__(self, input_filename, base_image_dir, img_key,
                 caption_key, clip_model: str, max_len: int = 77, sep="\t",
                 precision: str = 'fp32', transform=None, random_caption=False):
        super().__init__(input_filename, base_image_dir, img_key,
                         caption_key, clip_model, max_len,
                         sep, precision, transform, random_caption)

        self.embeddings = None

        # Whether or not to preprocess the nearest neighbor indices.
        # If `preprocess` is True, an nn indice array will be calculated,
        # which will be faster but allow multiple sampling of one instance.
        self.nn_indices = None
        del self.cache

    def get_synthesis_batch(self, batch_size, num_clusters: int = 1, max_sim: float = 1.0, neighborhood: int = 1):
        assert batch_size % num_clusters == 0, "Nonidentical clusters not supported currently."
        samples_per_cluster = batch_size // num_clusters

        if self.nn_indices is not None:
            assert neighborhood == 1
            indices = torch.randperm(self.__len__())[:num_clusters]
            selected_sample_indices = self.nn_indices[indices]
            selected_sample_indices = selected_sample_indices.flatten()

        else:
            selected_samples = np.zeros(self.__len__(), dtype=bool)  # Binary mask

            for i in range(num_clusters):
                unselected_indices = np.where(selected_samples == 0)[0]
                index = np.random.choice(unselected_indices)
                index_embedding = self.embeddings[index]

                # Calculate cosine similarity
                similarities = np.dot(self.embeddings, index_embedding) / (
                    np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(index_embedding))

                similarities[similarities >= max_sim] = -1.0
                similarities[selected_samples != 0] = -1.0  # Exclude existing samples
                similarities[index] = -1.0

                # Get indices of k highest similarities
                nearest_indices = np.argsort(similarities)[::-1][:samples_per_cluster * neighborhood]
                nearest_indices = np.random.choice(nearest_indices, size=samples_per_cluster - 1, replace=False)

                selected_samples[nearest_indices] = True
                selected_samples[index] = True

            selected_sample_indices = np.nonzero(selected_samples)[0]  # Convert to indices

        assert len(selected_sample_indices) == batch_size, (len(selected_sample_indices), batch_size) 
        batch = []
        for batch_idx, idx in enumerate(selected_sample_indices):
            batch.append(self.__getitem__(idx))

        return default_collate(batch)

    def update_embeddings(self, model, ordered_train_loader, modality='text', device='cuda'):
        assert modality in ['text', 'image']
        assert not isinstance(ordered_train_loader.sampler, torch.utils.data.RandomSampler)

        with torch.no_grad():
            embeddings = []
            for i, (_, images, tokens, _) in tqdm(enumerate(ordered_train_loader),
                                                  total=len(ordered_train_loader),
                                                  desc="Calculate sim matrix"):
                images, tokens = images.to(device), tokens.to(device)
                if modality == 'text':
                    batch_embeddings = model.encode_text(tokens)
                elif modality == 'image':
                    batch_embeddings = model.encode_image(images)
                
                embeddings.append(batch_embeddings.cpu())
            embeddings = torch.cat(embeddings)

            self.embeddings = embeddings

    def preprocess_neighbors(self, k_neighbors: int, batch_size: int = 10000):
        assert self.embeddings is not None

        normalized_embeddings = F.normalize(self.embeddings, p=2, dim=1)
        top_k_indices_all = []

        # Process each embedding one by one
        for i in tqdm(range(0, normalized_embeddings.size(0), batch_size),
                      desc="Preproces NNs"):
            # Compute cosine similarity of the current embedding with all others
            cosine_sim = torch.mm(
                normalized_embeddings[i: min(i + batch_size, self.__len__())],
                normalized_embeddings.t(),
            )

            # Get the top k+1 neighbors (including self)
            _, top_k_indices = torch.topk(cosine_sim, k=k_neighbors+1, dim=-1)
            top_k_indices = top_k_indices[:, :k_neighbors]

            top_k_indices_all.append(top_k_indices)

        self.nn_indices = torch.cat(top_k_indices_all)
        assert self.nn_indices.shape == (self.__len__(), k_neighbors), self.nn_indices.shape
