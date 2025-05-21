import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import *
from config import CHECKPOINT_PATH

def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        total_mpjpe = 0
        total_pck = 0
        total_oks = 0
        total_nme = 0
        total_failure_rate = 0
        total_kda = 0
        total_ap = 0
        total_ar = 0

        for images, keypoints, _, _ in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, keypoints[:, :, :2])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for images, keypoints, width, height in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)
                predictions = model(images)
                loss = criterion(predictions, keypoints[:, :, :2])
                total_val_loss += loss.item()

                pred_keypoints = predictions.cpu()
                gt_keypoints = keypoints[:, :, :2].cpu()
                width = torch.tensor(width, dtype=torch.float32)
                height = torch.tensor(height, dtype=torch.float32)
                scale = torch.sqrt(width * height).cpu().unsqueeze(1)
                normalization_factor = ((width + height) / 2).cpu().unsqueeze(1)

                total_mpjpe += calculate_mpjpe(pred_keypoints, gt_keypoints).item()
                total_pck += calculate_pck(pred_keypoints, gt_keypoints).item()
                total_oks += calculate_oks(pred_keypoints, gt_keypoints, scale).item()
                total_nme += calculate_nme(pred_keypoints, gt_keypoints, normalization_factor).item()
                total_failure_rate += calculate_failure_rate(pred_keypoints, gt_keypoints).item()
                total_kda += calculate_kda(pred_keypoints, gt_keypoints).item()
                ap, ar = calculate_ap_ar(pred_keypoints, gt_keypoints, scale)
                total_ap += ap
                total_ar += ar

        model.train()
        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"🔥 New best model saved with val_loss {best_val_loss:.6f}")

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {total_train_loss / len(train_loader):.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"MPJPE: {total_mpjpe / len(val_loader):.6f}, "
              f"PCK: {total_pck / len(val_loader):.6f}, "
              f"OKS: {total_oks / len(val_loader):.6f}, "
              f"NME: {total_nme / len(val_loader):.6f}, "
              f"Failure Rate: {total_failure_rate / len(val_loader):.6f}, "
              f"KDA: {total_kda / len(val_loader):.6f}, "
              f"AP: {total_ap / len(val_loader):.6f}, "
              f"AR: {total_ar / len(val_loader):.6f}")
