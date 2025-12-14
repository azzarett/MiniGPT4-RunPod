FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Model will be auto-downloaded on first run if not present
# To pre-download during build (optional, adds ~13GB to image):
# RUN mkdir -p checkpoints/mini-gpt4-7b && \
#     cd checkpoints/mini-gpt4-7b && \
#     wget https://huggingface.co/Vision-CAIR/MiniGPT-4/resolve/main/model.pth -O model.pth

# Set the entrypoint
CMD [ "python", "-u", "handler.py" ]
