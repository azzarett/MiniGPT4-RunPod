import os
import base64
import io
import torch
import runpod
import requests
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION_minigptv2

# Set HuggingFace cache to /runpod-volume/cache to use Network Volume
os.environ['HF_HOME'] = '/runpod-volume/cache'
os.environ['HF_HUB_CACHE'] = '/runpod-volume/cache/hub'
os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/cache/transformers'

print(f"HF_HOME set to: {os.environ.get('HF_HOME')}")

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
    cfg_path = os.environ.get("CFG_PATH", "eval_configs/minigptv2_eval.yaml")
    
    # Force correct path for MiniGPT-v2 to avoid conflict with old env vars
    default_v2_path = "/runpod-volume/checkpoints/minigptv2/checkpoint.pth"
    model_path = os.environ.get("MODEL_PATH", default_v2_path)
    
    # If the env var points to the old v1 model, force switch to v2 path
    if "mini-gpt4-7b" in model_path:
        print(f"Detected old model path in env: {model_path}. Switching to MiniGPT-v2 path: {default_v2_path}")
        model_path = default_v2_path

    model_url = "https://huggingface.co/spaces/Vision-CAIR/MiniGPT-v2/resolve/main/minigptv2_checkpoint.pth"
    print(f"Using Model URL: {model_url}")
    
    llama_model = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-2-7b-chat-hf")
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
    model_config.image_size = 448 # Increase resolution for document reading
    
    # Override llama model path (HF repo or local path)
    if llama_model:
        model_config.llama_model = llama_model
    
    # Ensure image size is 448 for v2
    model_config.image_size = 448

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    
    # Manually construct visual processor config since we disabled dataset builders
    vis_processor_cfg = OmegaConf.create({
        "name": "blip2_image_eval",
        "image_size": 448
    })
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    return model, vis_processor, args.gpu_id, "minigpt_v2"

print("Loading model...")
print(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/checkpoints/minigptv2/checkpoint.pth')}")
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
    'pretrain_llama2': CONV_VISION_LLama2,
    'minigpt_v2': CONV_VISION_minigptv2
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
                if "," in image_input:
                    image_input = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                print(f"Error decoding image: {e}")
                return {"error": "Invalid image data"}

        print(f"Image loaded: {image.size}")
        instruction = job_input.get("instruction", "Describe this image.")
        print(f"Instruction: {instruction}")
        
        # Prepare chat
        print("Initializing Chat object...")
        chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
        
        chat_state = conv_dict[model_type].copy()
        img_list = []
        
        print("Uploading image...")
        llm_message = chat.upload_img(image, chat_state, img_list)
        
        print("Encoding image...")
        chat.encode_img(img_list)
        
        print("Asking instruction...")
        chat.ask(instruction, chat_state)
        
        print(f"Prompt: {chat_state.get_prompt()}")

        print("Generating answer...")
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=5,
                                  do_sample=False,
                                  temperature=1.0,
                                  repetition_penalty=1.05,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        print("Answer generated.")
                                  
        return {"output": llm_message}
        
    except Exception as e:
        print(f"Handler exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
