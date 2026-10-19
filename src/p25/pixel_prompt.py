"""Trainable additive pixel-space border prompt for OpenAI CLIP.

Inputs and outputs are CLIP-normalized images (N, 3, 224, 224).
The perturbation is learned in RGB [0, 1] space on an 8-pixel border by default.
Only the prompt parameters require gradients; the CLIP backbone remains frozen.
"""

import torch
from torch import nn


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PixelBorderPrompt(nn.Module):
    def __init__(self, image_size=224, border_width=8, max_delta=0.25):
        super().__init__()
        if image_size <= 0 or not (0 < border_width * 2 < image_size):
            raise ValueError("Border width must be positive and less than half of image_size")
        if not (0 < max_delta <= 1):
            raise ValueError("max_delta must be between 0 and 1")
        self.image_size = image_size
        self.border_width = border_width
        self.max_delta = float(max_delta)

        # Same 3-channel spatial prompt is shared by every training image.
        # tanh() smoothly limits each RGB perturbation to +/-max_delta.
        self.raw_delta = nn.Parameter(torch.zeros(1, 3, image_size, image_size))
        mask = torch.ones(1, 1, image_size, image_size)
        mask[:, :, border_width:-border_width, border_width:-border_width] = 0
        self.register_buffer("mask", mask)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, normalized_images):
        if (normalized_images.ndim != 4 or normalized_images.shape[1:] !=
                (3, self.image_size, self.image_size)):
            raise ValueError(f"Expected [batch,3,{self.image_size},{self.image_size}]")
        rgb = normalized_images * self.std + self.mean
        delta = self.max_delta * torch.tanh(self.raw_delta) * self.mask
        prompted_rgb = (rgb + delta).clamp(0.0, 1.0)
        return (prompted_rgb - self.mean) / self.std

    def effective_trainable_parameters(self):
        # Count the border pixels, rather than the masked-out interior tensor.
        return int(self.mask.sum().item()) * 3
