import os
import base64
import io
import torch
import runpod
import requests
from PIL import Image
from pathlib import Path
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# Setup configuration
class Args:
    def __init__(self, cfg_path, gpu_id=0, options=None):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.options = options

def download_model(model_url, model_path):
    """Download model from HuggingFace if not already present."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    print(f"Downloading model from {model_url}...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Downloaded {percent:.1f}%", end='\r')
        
        print(f"\nModel downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            os.remove(model_path)
        raise

def setup_model():
    cfg_path = os.environ.get("CFG_PATH", "eval_configs/minigpt4_eval.yaml")
    model_path = os.environ.get("MODEL_PATH", "checkpoints/mini-gpt4-7b/model.pth")
    model_url = os.environ.get("MODEL_URL", "https://huggingface.co/Vision-CAIR/MiniGPT-4/resolve/main/model.pth")
    gpu_id = int(os.environ.get("GPU_ID", 0))
    
    # Auto-download model if not present
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        download_model(model_url, model_path)
    
    args = Args(cfg_path=cfg_path, gpu_id=gpu_id)
    cfg = Config(args)
    
    # Override checkpoint path if provided in env
    if model_path and os.path.exists(model_path):
        cfg.model_cfg.ckpt = model_path

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    return model, vis_processor, args.gpu_id, model_config.model_type

print("Loading model...")
print(f"Model path: {os.environ.get('MODEL_PATH', 'checkpoints/mini-gpt4-7b/model.pth')}")
# Initialize model globally to cache it
try:
    model, vis_processor, gpu_id, model_type = setup_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    # We don't exit here to allow the container to start, but handler will fail if model is not loaded
    model = None

conv_dict = {
    'pretrain_vicuna0': CONV_VISION_Vicuna0,
    'pretrain_llama2': CONV_VISION_LLama2
}

def handler(event):
    if model is None:
        return {"error": "Model failed to load during initialization."}

    try:
        job_input = event["input"]
        
        # Handle image input (URL or base64)
        image_input = job_input.get("image")
        if not image_input and "file" in job_input: # Support legacy 'file' key
             image_input = job_input["file"]
             
        if not image_input:
            return {"error": "No image provided"}

        if image_input.startswith("http"):
            image = Image.open(requests.get(image_input, stream=True).raw).convert('RGB')
        else:
            # Assume base64
            try:
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except:
                return {"error": "Invalid image data"}

        instruction = job_input.get("instruction", "Describe this image.")
        
        # Prepare chat
        # Re-initializing Chat object for each request to ensure clean state, 
        # but model and processor are reused.
        chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
        
        chat_state = conv_dict[model_type].copy()
        img_list = []
        
        llm_message = chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)
        
        chat.ask(instruction, chat_state)
        
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
                                  
        return {"output": llm_message}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
