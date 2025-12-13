from PIL import Image
from minigpt4 import MiniGPT4
import io
import base64

# Указываем путь к PyTorch чекпоинту
model = MiniGPT4("checkpoints/mini-gpt4-7b/model.pth")

def handler(event):
    try:
        image_bytes = base64.b64decode(event["file"])
        img = Image.open(io.BytesIO(image_bytes))
        instruction = event.get("instruction", "")
        result = model.generate(img, instruction)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
