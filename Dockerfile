FROM python:3.10-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем весь код
COPY . .

# Запускаемый файл не нужен, RunPod вызывает handler.py напрямую
