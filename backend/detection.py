from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/test")
def test(request: TextRequest): 
    processed_text = f"Обработано: {request.text}"

    print(processed_text)
    
    return {"text": processed_text}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)