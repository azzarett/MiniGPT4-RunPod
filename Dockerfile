FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr git wget build-essential python3-dev cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create volume directory structure
RUN mkdir -p /runpod-volume/cache /runpod-volume/checkpoints

# Set environment variables for cache
ENV HF_HOME="/runpod-volume/cache"
ENV HF_HUB_CACHE="/runpod-volume/cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/cache/transformers"

# Model will be auto-downloaded on first run if not present
# To pre-download during build (optional, adds ~13GB to image):
# RUN mkdir -p checkpoints/mini-gpt4-7b && \
#     cd checkpoints/mini-gpt4-7b && \
#     wget https://huggingface.co/Vision-CAIR/MiniGPT-4/resolve/main/model.pth -O model.pth

# Set the entrypoint
CMD [ "python", "-u", "handler.py" ]
