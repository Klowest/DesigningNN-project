import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

def load_yolo_annotations(img_path, label_path):
    """
    Загружает изображение и аннотации в формате YOLO.
    Возвращает: image [3, H, W], targets [M, 5] = [cls, x, y, w, h] (нормализованные)
    !!!ВОЗВРАЩАЕТ классы с индексами от 0, даже если на вход идут начиная с 1!!!
    """
    # Загрузка изображения
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Загрузка аннотаций
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        targets = torch.empty(0, 5)
    else:
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                cls_id, x, y, w, h = parts[:5]
                boxes.append([cls_id-1, x, y, w, h])
        targets = torch.tensor(boxes) if boxes else torch.empty(0, 5)

        if (targets[:, 0].min == 1):
            targets[:, 0] -= 1

    return img, targets

import os
import numpy as np
from PIL import Image
import cv2 

def build_rare_patches(img_paths, label_paths, imgsz, output_dir="patches", augment_classes={0, 1, 2, 4}):
    os.makedirs(output_dir, exist_ok=True)
    patches = []  # [(patch, mask, cls_id), ...]
    
    for idx, (img_path, label_path) in enumerate(zip(img_paths, label_paths)):
        img_pil, targets = load_yolo_annotations(img_path, label_path)
        img = np.array(img_pil)  # [H_orig, W_orig, 3], uint8
        H_orig, W_orig = img.shape[:2]

        img_resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        # img_resized = np.array(img_pil.resize((imgsz, imgsz), Image.LANCZOS))
        
        for target in targets:
            if target.numel() == 0:
                continue
            cls_id = int(target[0])
            if cls_id not in augment_classes:
                continue

            x, y, w, h = target[1:].tolist()

            cx, cy = x * imgsz, y * imgsz
            bw, bh = w * imgsz, h * imgsz
            x1_full = int(cx - bw / 2)
            y1_full = int(cy - bh / 2)
            x2_full = int(cx + bw / 2)
            y2_full = int(cy + bh / 2)

            x1_full = max(0, x1_full)
            y1_full = max(0, y1_full)
            x2_full = min(imgsz, x2_full)
            y2_full = min(imgsz, y2_full)

            if x2_full <= x1_full or y2_full <= y1_full:
                continue

            pad_x = int(0.1 * bw)
            pad_y = int(0.1 * bh)
            x1_pat = max(0, x1_full - pad_x)
            y1_pat = max(0, y1_full - pad_y)
            x2_pat = min(imgsz, x2_full + pad_x)
            y2_pat = min(imgsz, y2_full + pad_y)

            patch = img_resized[y1_pat:y2_pat, x1_pat:x2_pat]  # [hp, wp, 3]
            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                continue

            bbox_x1_in_pat = x1_full - x1_pat
            bbox_y1_in_pat = y1_full - y1_pat
            bbox_x2_in_pat = x2_full - x1_pat
            bbox_y2_in_pat = y2_full - y1_pat

            bbox_x1_in_pat = max(0, bbox_x1_in_pat)
            bbox_y1_in_pat = max(0, bbox_y1_in_pat)
            bbox_x2_in_pat = min(patch.shape[1], bbox_x2_in_pat)
            bbox_y2_in_pat = min(patch.shape[0], bbox_y2_in_pat)

            if bbox_x2_in_pat <= bbox_x1_in_pat or bbox_y2_in_pat <= bbox_y1_in_pat:
                continue

            mask = np.zeros(patch.shape[:2], dtype=bool)
            mask[bbox_y1_in_pat:bbox_y2_in_pat, bbox_x1_in_pat:bbox_x2_in_pat] = True

            # Сохраняем
            cls_name = {0: 'ball', 1: 'coach', 2: 'gk', 4: 'ref'}.get(cls_id, f'cls{cls_id}')
            path = os.path.join(output_dir, f"{cls_name}_{idx}_{len(patches)}.png")
            try:
                Image.fromarray(patch).save(path)
            except Exception as e:
                print(f"⚠️ Не удалось сохранить {path}: {e}")
                continue

            patches.append((patch, mask, cls_id))
    
    counts = {k: sum(1 for p in patches if p[2] == k) for k in augment_classes}
    print(f"Собрано {len(patches)} редких патчей: {counts}")
    return patches

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, patches_dir, imgsz=640, augment=True):
        self.img_paths = sorted([
            os.path.join(img_dir, f) 
            for f in os.listdir(img_dir) 
            if f.endswith(('.jpg', '.png'))
        ])
        
        self.label_paths = [
            os.path.join(label_dir, os.path.basename(p).replace('.jpg', '.txt').replace('.png', '.txt'))
            for p in self.img_paths
        ]
        self.imgsz = imgsz
        self.augment = augment

        # Статистика для oversampling
        self.class_counts = {
            0: 739,   # ball
            1: 202,   # coach
            2: 281,   # goalkeeper
            3: 12261, # player
            4: 1321   # referee
        }
        self.augment_classes = {0}

        max_count = max(self.class_counts.values())
        self.oversample_factors = {
            cls_id: min(5, max_count // count)
            for cls_id, count in self.class_counts.items()
            if cls_id in self.augment_classes and count > 0
        }

        self._rare_patches = None
        if (augment):
            self._rare_patches = build_rare_patches(self.img_paths, self.label_paths, self.imgsz, patches_dir)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        
        img, targets = load_yolo_annotations(img_path, label_path)  # PIL, torch.Tensor [N, 5]
        
        # Проверка: есть ли редкие классы в этом изображении
        has_rare = False
        if targets.numel() > 0:
            cls_ids = targets[:, 0].int()
            has_rare = bool((cls_ids.unsqueeze(1) == torch.tensor(list(self.augment_classes))).any())

        # Решаем: делать ли oversampling?
        do_oversample = self.augment and has_rare and random.random() < 0.7
        
        if do_oversample:
            img, targets = self._augment_with_copy_paste(img, targets)
        elif self.augment:
            img, targets = self._basic_augment(img, targets)
        else:
            img, targets = self._resize_and_normalize(img, targets)
    
        return img, targets  # img: torch.Tensor [3, H, W], targets: torch.Tensor [M, 5]
    
    def _clamp_bbox_coords(self, bboxes):
        """Приводит bbox к [0, 1] с учётом YOLO-формата: [x, y, w, h]"""
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return bboxes
        
        # x, y ∈ [0, 1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0.0, 1.0)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0.0, 1.0)
        
        # w, h ≥ 0 и таковы, что bbox не выходит за границы
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0.0, 1.0)  # w
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0.0, 1.0)  # h
        
        # Доп. защита: чтобы x ± w/2 ∈ [0,1]
        bboxes[:, 2] = np.minimum(bboxes[:, 2], 2 * np.minimum(bboxes[:, 0], 1 - bboxes[:, 0]))
        bboxes[:, 3] = np.minimum(bboxes[:, 3], 2 * np.minimum(bboxes[:, 1], 1 - bboxes[:, 1]))
    
        return bboxes

    def _basic_augment(self, img, targets):
        # Конвертируем targets в numpy для Albumentations
        # координаты остаются в формате YOLO
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()  # [N, 5]

        class_labels = targets[:, 0].astype(int).tolist() if len(targets) > 0 else []
        bboxes = targets[:, 1:].tolist() if len(targets) > 0 else []
        bboxes = self._clamp_bbox_coords(bboxes)

        transform = A.Compose([
            # A.Mosaic(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(height=self.imgsz, width=self.imgsz),
            # A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1,
            min_area=1.0
        ))

        transformed = transform(
            image=np.array(img),  # PIL → (H, W, 3), uint8, RGB
            bboxes=bboxes,
            class_labels=class_labels
        )

        # Восстановление targets как torch.Tensor
        if len(transformed['bboxes']) > 0:
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            classes = torch.tensor(transformed['class_labels'], dtype=torch.float32).unsqueeze(1)
            targets = torch.cat([classes, bboxes], dim=1)  # [M, 5]
        else:
            targets = torch.empty(0, 5, dtype=torch.float32)

        img_tensor = transformed['image'] / 255.0  # [3, H, W], float32, [0,1]
        return img_tensor, targets

    def _resize_and_normalize(self, img, targets):
        # координаты остаются в формате YOLO
        """Без аугментаций — только resize + ToTensor + нормализация."""
        transform = A.Compose([
            A.Resize(height=self.imgsz, width=self.imgsz),
            # A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        bboxes = targets[:, 1:].tolist() if len(targets) > 0 else []
        bboxes = self._clamp_bbox_coords(bboxes)
        class_labels = targets[:, 0].astype(int).tolist() if len(targets) > 0 else []

        transformed = transform(
            image=np.array(img),
            bboxes=bboxes,
            class_labels=class_labels
        )

        if len(transformed['bboxes']) > 0:
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            classes = torch.tensor(transformed['class_labels'], dtype=torch.float32).unsqueeze(1)
            targets = torch.cat([classes, bboxes], dim=1)
        else:
            targets = torch.empty(0, 5, dtype=torch.float32)

        return transformed['image'] / 255.0, targets

    def _find_safe_position(self, img, bboxes, patch_shape):
        """
            Случайный поиск свободного места
            Возвращает абсолютные x и y или -1 -1 если не удалось найти место
        """
        H, W = img.shape[:2]
        ph, pw = patch_shape[:2]
        max_tries = 200
        for _ in range(max_tries):
            x = random.randint(0, max(0, W - pw))
            y = random.randint(0, max(0, H - ph))
            in_range_box = False
            intersection = False
            # Проверка пересечения с существующими bbox (грубая)
            cx, cy = x + pw / 2, y + ph / 2
            for bx, by, bw, bh in bboxes:
                # Денормализуем bbox
                bx *= W; by *= H; bw *= W; bh *= H
                if abs(cx - bx) < (bw + pw)/2 and abs(cy - by) < (bh + ph)/2:
                    intersection = True
                    break  # пересекается
                if by - bh / 2 < cy < by + bh / 2:
                    in_range_box = True
            if (not intersection and in_range_box):
                return x, y
        return -1, -1

    def _paste_with_mask(self, img, patch, mask, x, y):
        """Вставка с маской (patch и mask — np.ndarray)."""
        h, w = patch.shape[:2]
        # Границы
        x1, x2 = x, min(x + w, img.shape[1])
        y1, y2 = y, min(y + h, img.shape[0])
        pw, ph = x2 - x1, y2 - y1

        # Маска может быть меньше из-за выхода за границы
        mask_roi = mask[:ph, :pw]
        patch_roi = patch[:ph, :pw]
        img_roi = img[y1:y2, x1:x2]

        # Альфа-смешение (можно упростить до замены)
        img_roi[mask_roi] = patch_roi[mask_roi]
        return img

    def _augment_with_copy_paste(self, img, targets):
        # Сначала базовые аугментации → получим изображение в [3, H, W], float32, [0,1]
        img_tensor, targets = self._basic_augment(img, targets)  # torch.Tensor

        # Конвертируем в numpy для вставки
        np_img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3], uint8
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

        bboxes = targets_np[:, 1:].copy() if len(targets_np) > 0 else np.empty((0, 4))
        classes = targets_np[:, 0].copy() if len(targets_np) > 0 else np.array([])

        for cls_id in self.augment_classes:
            n_paste = random.randint(1, self.oversample_factors.get(cls_id, 1))
            candidates = [p for p in self._rare_patches if p[2] == cls_id]
            if not candidates:
                continue
            for _ in range(n_paste):
                patch_img, patch_mask, _ = random.choice(candidates)
                x, y = self._find_safe_position(np_img, bboxes, patch_img.shape)
                if (x == -1 or y == -1):
                    continue
                np_img = self._paste_with_mask(np_img, patch_img, patch_mask, x, y)

                # Добавляем bbox в YOLO-формате (нормализованный)
                ph, pw = patch_img.shape[:2]
                cx = (x + pw / 2) / self.imgsz
                cy = (y + ph / 2) / self.imgsz
                bw = pw / self.imgsz
                bh = ph / self.imgsz

                bboxes = np.vstack([bboxes, [cx, cy, bw, bh]])
                classes = np.append(classes, cls_id)

        # Обратно в тензор
        img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # [3, H, W], [0,1]
        if len(classes) > 0:
            classes = torch.from_numpy(classes).float().unsqueeze(1)
            bboxes = torch.from_numpy(bboxes).float()
            targets = torch.cat([classes, bboxes], dim=1)  # [M, 5]
        else:
            targets = torch.empty(0, 5, dtype=torch.float32)

        return img_tensor, targets