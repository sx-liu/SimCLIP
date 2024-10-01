import os
import math
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import CLIP
from models.data import CLIPDataset
from models.utils import clip_loss, accuracy
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

import wandb


class CLIPWrapper(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrain: bool = False,
                 lr: float = None,
                 total_steps: int = None,
                 data_parallel: bool = False,
                 device_ids: list = None
                 ):
        super().__init__()

        self.model_name = model_name
        self.model = CLIP(model_name, pretrain)
        self.data_parallel = data_parallel

        if self.data_parallel:
            print('Enabling Data parallel.')
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))

            print(f'Using cuda devices: {device_ids}')

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if lr is None:
            lr = 5e-4 if pretrain else 1e-5

        if pretrain:
            assert total_steps is not None
            self.optimizer, self.lr_scheduler = self.configure_optimizers(5e-4, "cosine", total_steps)
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def configure_optimizers(self, lr=1e-5, scheduler_type: str = None, total_steps: int = None):
        if scheduler_type is None:
            scheduler_type = "constant"

        assert scheduler_type in ["constant", "cosine"]
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.1
        )

        # scheduler_steplr = StepLR(optimizer, step_size=20000, gamma=0.1)
        if scheduler_type == "constant":
            # Gradual warm up with constant schedular
            lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=200,
                                                  after_scheduler=None)
        else:
            # Source: https://github.com/openai/CLIP/issues/107
            # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=total_steps,
                cycle_mult=1.0,
                max_lr=lr,
                min_lr=0,
                warmup_steps=2000
            )

        return optimizer, lr_scheduler

    def encode_image(self, inputs):
        if self.data_parallel:
            return self.model.module.encode_image(inputs)
        else:
            return self.model.encode_image(inputs)

    def encode_text(self, inputs):
        if self.data_parallel:
            return self.model.module.encode_text(inputs)
        else:
            return self.model.encode_text(inputs)

    def training_step(self, train_batch, step, weights=None):
        # get optimizers and scheduler
        self.train()
        self.optimizer.zero_grad()

        images, tokens = train_batch
        image_embeddings, text_embeddings = self.model(images, tokens)

        if self.data_parallel:
            logits_per_image, logits_per_text = self.model.module.calculate_logits(image_embeddings, text_embeddings)
        else:
            logits_per_image, logits_per_text = self.model.calculate_logits(image_embeddings, text_embeddings)
        loss = clip_loss(logits_per_image, logits_per_text, weights=weights)

        loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()

        if self.data_parallel:
            self.model.module.logit_scale.data.clamp_(-np.log(100), np.log(100))
        else:
            self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

        with torch.no_grad():
            i2t_acc = accuracy(logits_per_image)
            t2i_acc = accuracy(logits_per_text)

        self.log('train/loss', loss.item(), step)
        self.log('train/i2t', i2t_acc, step)
        self.log('train/t2i', t2i_acc, step)
        return loss.item(), i2t_acc, t2i_acc

    def validation_step(self, val_batch, step):
        self.eval()
        images, tokens = val_batch
        with torch.no_grad():
            image_embeddings, text_embeddings = self.model(images, tokens)

            if self.data_parallel:
                logits_per_image, logits_per_text = self.model.module.calculate_logits(image_embeddings, text_embeddings)
            else:
                logits_per_image, logits_per_text = self.model.calculate_logits(image_embeddings, text_embeddings)
                
            loss = clip_loss(logits_per_image, logits_per_text)

            i2t_acc = accuracy(logits_per_image)
            t2i_acc = accuracy(logits_per_text)

        self.log('val/loss', loss.item(), step)
        self.log('val/i2t', i2t_acc, step)
        self.log('val/t2i', t2i_acc, step)
        return loss.item(), i2t_acc, t2i_acc

    def calculate_emb(self, val_batch):
        self.eval()
        image, text = val_batch
        with torch.no_grad():
            if self.data_parallel:
                image_emb = F.normalize(self.model.module.encode_image(image), dim=1)
                text_emb = F.normalize(self.model.module.encode_text(text), dim=1)
            else:
                image_emb = F.normalize(self.model.encode_image(image), dim=1)
                text_emb = F.normalize(self.model.encode_text(text), dim=1)

        return image_emb, text_emb

    def save_model(self, model_name, save_dir, **kwargs):
        save_dir = os.path.join(save_dir, model_name)

        if self.data_parallel:
            # torch.save(self.model.module.state_dict(), save_dir + '.pth')
            self.model.module.pretrained_clip_model.save_pretrained(save_dir)
        else:
            # torch.save(self.model.state_dict(), save_dir + '.pth')
            self.model.pretrained_clip_model.save_pretrained(save_dir)
            
        if kwargs:
            with open(os.path.join(save_dir, "model_args.json"), 'w') as json_file:
                json.dump(kwargs, json_file)

    @staticmethod
    def log(item, value, step):
        wandb.log({item: value}, step=step)
