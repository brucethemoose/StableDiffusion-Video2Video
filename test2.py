import torch
from diffusers import StableDiffusionImg2ImgPipeline
import utils.images2img
from torchvision.utils import save_image
from torchvision.io import read_image
from PIL import Image
import os
import torch._dynamo as dynamo

img = Image.open("/home/alpha/tensorsave.png")
img2 = Image.open("/home/alpha/tensorsave.png")
lat = torch.randn([1,4,64,64], dtype = torch.float16, device="cuda")

img = img.convert("RGB")
img2 = img2.convert("RGB")

# make sure you're logged in with `huggingface-cli login`
pipe = utils.images2img.StableDiffusionImages2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
image = pipe("4K, dark", latents=lat, image = img, image2 = img2, strength=0.01, num_inference_steps = 20).images[0]
os.remove("/home/alpha/test2.png")
image.save("/home/alpha/test2.png")
