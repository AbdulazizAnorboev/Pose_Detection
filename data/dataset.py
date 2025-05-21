import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.transform = transform
        self.images = {img["id"]: img for img in self.data["images"]}
        self.annotations = self.data["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann["image_id"]]
        image_path = os.path.join(self.image_root, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        keypoints = torch.tensor(ann["keypoints"], dtype=torch.float32).view(-1, 3)
        keypoints[:, 0] /= image_info["width"]
        keypoints[:, 1] /= image_info["height"]
        return image, keypoints, image_info["width"], image_info["height"]
