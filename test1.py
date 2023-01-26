import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

img = Image.open("/home/alpha/tensorsave.png")
img2 = Image.open("/home/alpha/tensorsave.png")
lat = torch.randn([1,4,64,64], dtype = torch.float16, device="cuda")

img = img.convert("RGB")
img2 = img2.convert("RGB")

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
image = pipe("4K, dark", image = img,  strength=0.8, num_inference_steps = 20).images[0]
os.remove("/home/alpha/test2.png")
image.save("/home/alpha/test2.png")
