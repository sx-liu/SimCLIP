import torch
import torch.nn.functional as F


def clip_loss(logits_per_image, logits_per_text, alpha=0.0, weights=None):
    num_samples = logits_per_image.shape[0]
    if weights is None:
        weights = torch.ones(num_samples,
                             device=logits_per_image.device,
                             dtype=torch.float)

    labels = torch.arange(num_samples,
                          device=logits_per_image.device,
                          dtype=torch.long)

    total_loss = (F.cross_entropy(logits_per_image,
                                  labels,
                                  label_smoothing=alpha,
                                  weight=weights) +
                  F.cross_entropy(logits_per_text,
                                  labels,
                                  label_smoothing=alpha,
                                  weight=weights)) / 2

    return total_loss


def accuracy(logits, topk=1):
    labels = torch.arange(logits.shape[0],
                          device=logits.device,
                          dtype=torch.long)
    labels = labels.view(-1, 1)

    if topk > logits.shape[0]:
        topk = logits.shape[0]
        print(f'Use topk = {topk} instead')

    _, predictions = torch.topk(logits, k=topk, dim=1)
    result = (predictions == labels).any(dim=1)
    acc = torch.mean(result.float())

    return acc


def calculate_similarity_matrix(embeddings):
    # Normalize the embeddings to unit vectors
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms
    
    # Calculate the cosine similarity matrix
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
    
    return similarity_matrix


def find_top_k_nns(similarity_matrix, k=1):
    # Set diagonal to -1 to exclude self-similarity
    similarity_matrix.fill_diagonal_(-1)
    
    # Get the indices of the top k nearest neighbors for each sample
    _, nearest_neighbors = torch.topk(similarity_matrix, k, dim=1, largest=True, sorted=True)
    
    return nearest_neighbors


def get_data_from_indices(dataset, indices):
    # Extract the data using the indices tensor
    data_subset = [dataset[i] for i in indices]
    
    return default_collate(data_subset)
