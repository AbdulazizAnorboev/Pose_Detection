import torch

def calculate_mpjpe(pred_keypoints, gt_keypoints):
    return torch.mean(torch.norm(pred_keypoints - gt_keypoints, dim=-1))

def calculate_pck(pred_keypoints, gt_keypoints, threshold=0.2):
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    return torch.mean((distances < threshold).float()) * 100

def calculate_oks(pred_keypoints, gt_keypoints, scale, kappa=0.5):
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    scale = scale.unsqueeze(1)
    oks = torch.exp(-(distances ** 2) / (2 * (scale ** 2) * (kappa ** 2)))
    return torch.mean(oks)

def calculate_nme(pred_keypoints, gt_keypoints, normalization_factor):
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    return torch.mean(distances / normalization_factor)

def calculate_failure_rate(pred_keypoints, gt_keypoints, threshold=0.5):
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    oks = torch.exp(-(distances ** 2) / (2 * (threshold ** 2)))
    return torch.mean((oks < threshold).float()) * 100

def calculate_kda(pred_keypoints, gt_keypoints, threshold=5):
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    return torch.mean((distances < threshold).float()) * 100

def calculate_ap_ar(pred_keypoints, gt_keypoints, scale, thresholds=[0.50, 0.75]):
    oks_values = []
    for threshold in thresholds:
        oks = calculate_oks(pred_keypoints, gt_keypoints, scale)
        oks_values.append((oks > threshold).float().mean().item())
    ap = sum(oks_values) / len(oks_values)
    ar = max(oks_values)
    return ap * 100, ar * 100
