FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY ./backend/requirements.txt /app
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
COPY ./backend /app

CMD ["python3", "detection.py"]