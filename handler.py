from PIL import Image
from minigpt4 import MiniGPT4
import io
import base64
import os

# Путь к модели через env
model_path = os.environ.get("MODEL_PATH", "checkpoints/mini-gpt4-7b/model.pth")
model = MiniGPT4(model_path)


def handler(event, context):
    try:
        image_bytes = base64.b64decode(event["file"])
        img = Image.open(io.BytesIO(image_bytes))
        instruction = event.get("instruction", "")
        result = model.generate(img, instruction)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
