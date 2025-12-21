import torch
import math

import torch

def bbox_iou(pred_boxes, target_boxes):
    """
    Вычисляет IoU матрицу между боксами.
        pred_boxes: [N, 4] — (cx, cy, w, h)
        target_boxes: [M, 4] — (cx, cy, w, h)
    
    Returns:
        iou: [N, M]
    """
    # перевод в координаты (x1, y1, x2, y2)
    pred_xyxy = torch.cat([
        pred_boxes[:, :2] - pred_boxes[:, 2:] / 2,
        pred_boxes[:, :2] + pred_boxes[:, 2:] / 2
    ], dim=1).unsqueeze(1)  # [N, 1, 4]

    tgt_xyxy = torch.cat([
        target_boxes[:, :2] - target_boxes[:, 2:] / 2,
        target_boxes[:, :2] + target_boxes[:, 2:] / 2
    ], dim=1).unsqueeze(0)  # [1, M, 4]

    # Поиск пересечений
    lt = torch.max(pred_xyxy[..., :2], tgt_xyxy[..., :2])   # [N, M, 2]
    rb = torch.min(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])   # [N, M, 2]
    wh = (rb - lt).clamp(min=0)      # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    # вычисление iou    
    area_pred = (pred_boxes[:, 2] * pred_boxes[:, 3]).unsqueeze(1)     # [N, 1]
    area_tgt = (target_boxes[:, 2] * target_boxes[:, 3]).unsqueeze(0)  # [1, M]
    union = area_pred + area_tgt - inter  # [N, M]
    iou = inter / union.clamp(min=1e-7)

    return iou  # [N, M]

import torch
import math

def bbox_ciou(pred_boxes, target_boxes):
    """
    Вычисляет CIoU матрицу между боксами.
    
    Args:
        pred_boxes: [N, 4] — (cx, cy, w, h)
        target_boxes: [M, 4] — (cx, cy, w, h)
    
    Returns:
        ciou_loss: [N, M] — CIoU
    """

    # перевод в координаты (x1, y1, x2, y2)
    pred_xyxy = torch.cat([
        pred_boxes[:, :2] - pred_boxes[:, 2:] / 2,
        pred_boxes[:, :2] + pred_boxes[:, 2:] / 2
    ], dim=1).unsqueeze(1)  # [N, 1, 4]

    tgt_xyxy = torch.cat([
        target_boxes[:, :2] - target_boxes[:, 2:] / 2,
        target_boxes[:, :2] + target_boxes[:, 2:] / 2
    ], dim=1).unsqueeze(0)  # [1, M, 4]

    # поиск пересечений
    lt = torch.max(pred_xyxy[..., :2], tgt_xyxy[..., :2])   # [N, M, 2]
    rb = torch.min(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])   # [N, M, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    # вычисление iou    
    area_pred = (pred_boxes[:, 2] * pred_boxes[:, 3]).unsqueeze(1)     # [N, 1]
    area_tgt = (target_boxes[:, 2] * target_boxes[:, 3]).unsqueeze(0)  # [1, M]
    union = area_pred + area_tgt - inter
    iou = inter / union.clamp(min=1e-7)  # [N, M]

    # рассчёт диагонали наименьшего охватывающего прямоугольника
    c_lt = torch.min(pred_xyxy[..., :2], tgt_xyxy[..., :2])   # [N, M, 2]
    c_rb = torch.max(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])   # [N, M, 2]
    c_wh = (c_rb - c_lt).clamp(min=0)                         # [N, M, 2]
    c2 = c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2 + 1e-7         # [N, M]

    # расстояние между центрами
    pred_centers = pred_boxes[:, :2].unsqueeze(1)  # [N, 1, 2]
    tgt_centers = target_boxes[:, :2].unsqueeze(0)  # [1, M, 2]
    rho2 = ((pred_centers - tgt_centers) ** 2).sum(dim=2)  # [N, M]

    # рассчёт альфы
    pred_w, pred_h = pred_boxes[:, 2].unsqueeze(1), pred_boxes[:, 3].unsqueeze(1)  # [N, 1]
    tgt_w, tgt_h = target_boxes[:, 2].unsqueeze(0), target_boxes[:, 3].unsqueeze(0)  # [1, M]

    ar_pred = pred_w / (pred_h + 1e-7)  # [N, 1]
    ar_tgt = tgt_w / (tgt_h + 1e-7)     # [1, M]

    v = (4.0 / (math.pi ** 2)) * torch.pow(
        torch.atan(ar_tgt) - torch.atan(ar_pred), 2
    )  # [N, M]

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)  # [N, M]

    # рассчёт CIoU 
    ciou = iou - (rho2 / c2 + alpha * v)  # [N, M]
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    return ciou  # [N, M]