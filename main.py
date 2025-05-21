from data.dataset import PoseDataset
from models.model import ViTPose
from config import transform, BATCH_SIZE
from train import train
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

train_dataset = PoseDataset("train/_annotations.coco.json", "train", transform)
val_dataset = PoseDataset("valid/_annotations.coco.json", "valid", transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ViTPose().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

train(model, train_loader, val_loader, criterion, optimizer, epochs=20)
