from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import os
import uuid
import random
import tempfile
from io import BytesIO
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models import ResNet50_Weights

# КОНФИГУРАЦИЯ МОДЕЛИ
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/app/model/runs/weights/fasterrcnn_football_coco.pt"
SCORE_THRESHOLD = 0.6

LABELS = {
    1: "ball",
    2: "coach",
    3: "goalkeeper",
    4: "player",
    5: "referee",
}

COLORS = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
}

# Параметры обработки видео
FRAME_SKIP = 5       # Обрабатывать каждый 5-й кадр (пока так, чтобы долго не было)
CACHE_FRAMES = True  # Использовать последние боксы для промежуточных кадров (чтобы скачков не было)

# -----------------------------------
# Загрузка весов модели из MODEL_PATH
# ------------------------------------
def load_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=len(LABELS) + 1 
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    print("Model loaded")

    return model

# Сразу загружаем модель, чтобы потом не возиться
MODEL = load_model()

# -------------------------------------------------------
# ЗАГЛУШКА: имитация модели (это нужно, если нет весов)
# --------------------------------------------------------
def mock_detect(image: np.ndarray) -> List[Dict[str, Any]]:
    """Возвращает случайные боксы в формате [x1, y1, x2, y2]"""
    h, w = image.shape[:2]
    boxes = []
    for _ in range(random.randint(0, 5)):
        cls_id = random.choice(list(LABELS.keys()))
        conf = round(random.uniform(0.5, 0.99), 2)
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + 20, min(x1 + 200, w))
        y2 = random.randint(y1 + 20, min(y1 + 200, h))
        boxes.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
    return boxes

# -------------------------------------
# РЕАЛЬНЫЙ ДЕТЕКТ: модель из resnet50
# ------------------------------------
def detect_with_model(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    image: np.ndarray (H, W, 3), RGB
    """
    img_tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL([img_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    detections = []

    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.astype(int)

        detections.append({
            "class_id": int(label),
            "confidence": float(score),
            "bbox": [x1, y1, x2, y2],
        })

    return detections

# --------------------------------------
# ОТРИСОВКА БОКСОВ НА КАДРЕ (OpenCV)
# --------------------------------------
def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det["class_id"]
        conf = det["confidence"]
        label = f"{LABELS.get(cls_id, 'unknown')} {conf:.2f}"
        color = COLORS.get(cls_id, (255, 255, 255))

        # Рисуем бокс
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Подпись
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

# ----------------------------
# ОБРАБОТКА ВИДЕО
# ----------------------------
def process_video(video_bytes: bytes, ext: str) -> bytes:
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
        tmp_in.write(video_bytes)
        input_path = tmp_in.name

    # Выходной файл
    output_path = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")

    # Параметры видео извлекаем
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")

    # Выбираем кодек (mp4v для .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError("Не удалось создать VideoWriter")

    frame_idx = 0
    last_detections: List[Dict[str, Any]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем каждый FRAME_SKIP кадр  
        if frame_idx % FRAME_SKIP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # detections = mock_detect(frame) # когда использовалась заглушка
            detections = detect_with_model(rgb_frame)
            
            if CACHE_FRAMES or len(detections) > 0:
                last_detections = detections
            print(f"Кадр {frame_idx}: найдено {len(detections)} объектов")
        else:
            # Используем последние боксы (если они есть)
            detections = last_detections if CACHE_FRAMES else []

        # Наносим аннотации
        annotated_frame = draw_detections(frame.copy(), detections)
        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Читаем результат в байты
    with open(output_path, "rb") as f:
        result_bytes = f.read()

    # Удаляем временные файлы
    os.unlink(input_path)
    os.unlink(output_path)

    return result_bytes

# ----------------------------
# FASTAPI
# ----------------------------
app = FastAPI(title="Detection API", version="1.0")

# Проверка состояния
@app.get("/health")
def health():
    return {
        "status": "OK",
        "device": DEVICE,
        "model": "fasterrcnn_resnet50_fpn (football)"
    }

# Основная функция для детекции фото или видео
@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        name, ext = os.path.splitext(file.filename)
        ext = ext.lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            # ИЗОБРАЖЕНИЕ

            image = Image.open(BytesIO(contents)).convert("RGB")
            np_img = np.array(image)

            # detections = mock_detect(np_img)     # когда использовалась заглушка
            detections = detect_with_model(np_img)

            # Отрисовка через PIL
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except OSError:
                font = ImageFont.load_default()

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cls_id = det["class_id"]
                conf = det["confidence"]
                label = f"{LABELS.get(cls_id, 'unknown')} {conf:.2f}"
                color_rgb = COLORS.get(cls_id, (255, 255, 255))
                # OpenCV использует BGR, PIL - RGB поэтом конвертируем
                color_pil = tuple(reversed(color_rgb)) if isinstance(color_rgb, tuple) else color_rgb

                draw.rectangle([x1, y1, x2, y2], outline=color_pil, width=2)
                text_size = draw.textbbox((0, 0), label, font=font)
                text_w = text_size[2] - text_size[0]
                text_h = text_size[3] - text_size[1]
                draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color_pil)
                draw.text((x1 + 2, y1 - text_h - 2), label, fill=(0, 0, 0), font=font)

            output = BytesIO()
            image.save(output, format="JPEG" if ext in [".jpg", ".jpeg"] else "PNG")
            output.seek(0)
            new_name = f"detected_{name}_{str(uuid.uuid4())[:8]}{ext}"
            media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

            return StreamingResponse(
                output,
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{new_name}"'}
            )

        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # ВИДЕО
            print(f"Получено видео: {file.filename}")
            processed_video_bytes = process_video(contents, ext)

            new_name = f"detected_{name}_{str(uuid.uuid4())[:8]}.mp4"  # всегда .mp4 на выходе
            return StreamingResponse(
                BytesIO(processed_video_bytes),
                media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{new_name}"'}
            )

        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только .jpg, .png, .mp4, .avi, .mov, .mkv")

    except Exception as e:
        print(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)