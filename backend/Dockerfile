FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./backend/requirements.txt /app
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./backend /app

CMD ["python3", "detection.py"]