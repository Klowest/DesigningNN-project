from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import os
import uuid
from io import BytesIO

app = FastAPI(title="Detection API", version="1.0")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        # Читаем содержимое
        contents = await file.read()
        name, ext = os.path.splitext(file.filename)
        new_name = f"processed_{name}_{str(uuid.uuid4())[:8]}{ext}"

        # В будущем: здесь будет обработка нейросетью
        # Сейчас просто возвращаем тот же файл

        return StreamingResponse(
            BytesIO(contents),
            media_type=file.content_type,
            headers={"Content-Disposition": f'attachment; filename="{new_name}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)