from fastapi import FastAPI, UploadFile
from PIL import Image
from minigpt4 import MiniGPT4

app = FastAPI()
model = MiniGPT4("path/to/checkpoint")


@app.post("/analyze")
async def analyze(file: UploadFile, instruction: str):
    img = Image.open(file.file)
    response = model.generate(img, instruction)
    return {"result": response}
