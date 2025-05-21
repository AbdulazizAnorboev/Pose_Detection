import torch.nn as nn
from transformers import ViTModel

class ViTPose(nn.Module):
    def __init__(self, num_keypoints=6):
        super(ViTPose, self).__init__()
        self.vit = ViTModel.from_pretrained("facebook/dino-vitb16")
        self.mlp_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 2)
        )

    def forward(self, images):
        features = self.vit(images).last_hidden_state[:, 0, :]
        keypoints = self.mlp_head(features)
        return keypoints.view(-1, 6, 2)
