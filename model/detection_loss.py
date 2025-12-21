import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, complete_box_iou, box_iou


def detection_loss(pred, targets, pos_weight, num_classes, imgsz, device, topk=10, weight_box=5.0, weight_obj=1.0, weight_cls=1.0):
    """
    Лосс по предиктным боксам.
    Для всех предиктов проходит loss BCE по тому, является ли он объектом.
    А предикт является объектом, если он находится в topk по IoU у любого таргета и его максимальный IoU среди таргетов больше порога.
    Ошибка класса и бокса считается только для объектных предиктов.
    При этом предикту будет соответствовать таргет с наибольшим IoU.
    Для классов считается loss BCE, для боксов loss CIoU.
    
    Args:
        pred: [B, N, nc+1] = [cx, cy, w, h, obj, cls_1..cls_nc]
        targets: List[Tensor[M_i, 5]] = [class_id (0..4), cx, cy, w, h]
        num_classes: количество классов без учёта objects
        imgsz: размер изображения (должно быть imgsz x imgsz)

    Returns:
        total_loss, logs
    """
    B, N, D = pred.shape

    box_pred = pred[..., :4]    # [B, N, 4]
    obj_pred = pred[..., 4]     # [B, N]
    cls_pred = pred[..., 5:]    # [B, N, NC]

    loss_box = torch.tensor(0.0, device=device)
    loss_obj = torch.tensor(0.0, device=device)
    loss_cls = torch.tensor(0.0, device=device)
    total_pos = 0

    for b in range(B):
        tgt = targets[b].clone()  # [M, 5]
        if tgt.numel() == 0 or tgt.shape[0] == 0:
            # Только фон
            obj_target = torch.zeros(N, device=device)
            loss_obj += F.binary_cross_entropy_with_logits(
                obj_pred[b], obj_target, reduction='sum'
            )
            continue
            
        box_tgt = tgt[:, 1:] * imgsz

        # Конвертируем в xyxy (обязательно!)
        pred_xyxy = box_convert(box_pred[b], in_fmt="cxcywh", out_fmt="xyxy")
        tgt_xyxy = box_convert(box_tgt, in_fmt="cxcywh", out_fmt="xyxy")

        # Считаем CIoU matrix [N, M]
        # ciou_mat = complete_box_iou(pred_xyxy, tgt_xyxy)  # значения в [-1, 1]
        iou_mat = box_iou(pred_xyxy, tgt_xyxy)

        # Для каждого таргета — top-k предиктов по IoU
        topk_ious, topk_indices = torch.topk(iou_mat, k=min(topk, N), dim=0)  # [k, M]

        # Собираем все candidate предикты
        candidate_mask = torch.zeros(N, dtype=torch.bool, device=device)
        candidate_indices = topk_indices.flatten()  # [k*M]
        candidate_mask[candidate_indices] = True

        # также должен быть IoU выше порога хотя бы для одного таргета
        max_iou, assigned_tgt_idx = iou_mat.max(dim=1)    # [N]
        assigned_mask = candidate_mask & (max_iou > 0.2)

        # loss класса и бокса для живых боксов (obj_target != 0)
        pos_idx = torch.where(assigned_mask)[0]

        # loss для логита, который говорит о наличии объекта
        # то есть устремляем боксы с пересечением к 1, а без пересечения к 0
        # obj_target = assigned_mask.float()  # [N]
        obj_target = torch.zeros(N, device=device)
        obj_target[pos_idx] = max_iou[pos_idx].clamp(0.0, 1.0)
        loss_obj += F.binary_cross_entropy_with_logits(
            obj_pred[b], obj_target, reduction='sum'
        )

        if len(pos_idx) > 0:
            # это ближайшие по IoU таргеты, соответствующие живым боксам
            # ближайший для живого бокса таргет
            assigned_targets_idx = assigned_tgt_idx[pos_idx]
            assigned_targets = tgt[assigned_targets_idx]   # [P, NC]
            pred_boxes_pos = pred_xyxy[pos_idx]            # [P, 4]
            tgt_boxes_pos = tgt_xyxy[assigned_targets_idx] # [P, 4]
            tgt_classes = assigned_targets[:, 0].long()    # [P]

            # loss боксов Можно оптимизировать и считать только между элементами
            loss_box += (1 - complete_box_iou(pred_boxes_pos, tgt_boxes_pos).diag()).sum()

            # loss классов
            cls_target = torch.zeros(len(pos_idx), num_classes, device=device)
            cls_target[torch.arange(len(pos_idx)), tgt_classes] = 1.0
            loss_cls += F.binary_cross_entropy_with_logits(
                cls_pred[b][pos_idx], cls_target, reduction='sum', pos_weight=pos_weight
            )
            total_pos += len(pos_idx)

    # Нормировка
    loss_box = loss_box / max(total_pos, 1)
    loss_obj = loss_obj / (B * N)
    loss_cls = loss_cls / max(total_pos, 1)

    total_loss = weight_box * loss_box + weight_obj * loss_obj + weight_cls * loss_cls

    return total_loss, {
        "loss_box": loss_box.item(),
        "loss_obj": loss_obj.item(),
        "loss_cls": loss_cls.item(),
        "num_pos": total_pos
    }