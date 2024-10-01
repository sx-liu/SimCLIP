import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

# from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import CLIPModel, AutoConfig

class CLIP(nn.Module):
    def __init__(self, clip_version, pretrain=False):
        super().__init__()
        if pretrain:
            config = AutoConfig.from_pretrained(clip_version)
            self.pretrained_clip_model = CLIPModel(config=config)
        else:
            self.pretrained_clip_model = CLIPModel.from_pretrained(clip_version)

        # Retrieve image-side models
        self.vision_model = self.pretrained_clip_model.vision_model
        self.visual_projection = self.pretrained_clip_model.visual_projection

        # Retrieve text-side models
        self.text_model = self.pretrained_clip_model.text_model
        self.text_projection = self.pretrained_clip_model.text_projection

        # Logit scale
        # Logit scale has initial value nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = self.pretrained_clip_model.logit_scale

    def encode_text(self, inputs):
        embeddings = self.text_model(inputs).pooler_output
        embeddings = self.text_projection(embeddings)
        return embeddings

    def encode_image(self, inputs):
        embeddings = self.vision_model(inputs).pooler_output
        embeddings = self.visual_projection(embeddings)
        return embeddings

    def forward(self, images, text):
        image_embeddings = F.normalize(self.encode_image(images), dim=1)
        text_embeddings = F.normalize(self.encode_text(text), dim=1)
        
        return image_embeddings, text_embeddings
    
    def calculate_logits(self, image_embeddings, text_embeddings):
        logits_per_image = image_embeddings @ text_embeddings.T * self.logit_scale.exp()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
