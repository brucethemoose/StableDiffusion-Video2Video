import torch
from diffusers import StableDiffusionImg2ImgPipeline



import vapoursynth as vs
from vapoursynth import core



#Nasty fix because VSEdit cant import files outside of site-packages or the vapoursynth script folder for some reason
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
sys.path.append(os.path.dirname(__file__))
import utils.inference
import utils.images2img


#Make the stable diffusion pipe
lat = torch.randn([1,4,64,64], dtype = torch.float16, device="cuda")
pipe = utils.images2img.StableDiffusionImages2ImgPipeline.from_pretrained(pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16", safety_checker=None, requires_safety_checker=False)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

#Colorspace matrix to use for RGB <-> YUV conversion
matrix = "709"
#Get video input. Crop and resize input for easy testing, convert to FP32 YUV for BM3D
clip = core.ffms2.Source("/home/alpha/Storage/TrainingData/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m")
clip = core.std.Crop(clip, left = 968, right = 968)

#Denoise for more temporal consistency
r = 2 #temporal radius
s = 5 #denoising strength
clip = core.resize.Spline36(clip, format = vs.YUV444PS, width=512, height=512) # convert to floating point YUV for bm3d
vbasic = core.bm3dcpu.BM3Dv2(clip, radius=r-1, sigma = s, chroma = True)
clip = core.bm3dcpu.BM3Dv2(clip, ref=vbasic, radius=r, sigma = s, chroma = True)

#Generate motion compensated frame.
#TODO: Implement better compensation with another tool (SVP?). Initial testing with MVTools is not generating great results.
clip = core.resize.Point(clip, format = vs.YUV444P16) #Convert to 16-bit integer YUV for MVTools
super = core.mv.Super(clip, hpad=0, vpad=0)
back = core.mv.Analyse(super, isb=True, blksize=16, overlap=8, search=3, truemotion=True)
back = core.mv.Recalculate(super, back, blksize=8, overlap=4, search=3, truemotion=True)
forward = core.mv.Analyse(super, isb=False, blksize=16, overlap=8, search=3, truemotion=True)
forward = core.mv.Recalculate(super, forward, blksize=8, overlap=4, search=3, truemotion=True)
depanest = core.mv.DepanAnalyse(clip, forward, zoom=True, rot=True)
#depanest = core.mv.DepanEstimate(clip, trust = 5.0, zoommax = 2.0)
clip = core.mv.SCDetection(clip, back) # Detect scene changes using the motion estimates

# get previous frames
forward_comp = utils.inference.timemachine(clip)
forward_comp = core.resize.Point(forward_comp, format = clip.format, matrix_s = matrix) #Convert from RGB numpy to YUV
#Motion compensate moving objects
forward_comp = core.mv.Compensate(forward_comp, super, back)
#Motion compensate the whole frame, and offset the depan clip by 1 frame so that the "predicted" frames match the clip's frames
forward_comp = core.mv.DepanCompensate(forward_comp, depanest, offset=1, mirror = 15, subpixel=2, blur = 30)[1:] + clip[-1]



#forward_comp = core.text.Text(forward_comp, "forward")
#clip = core.std.Interleave([clip, forward_comp])

#Define SD processing function
def diffuse(img: torch.Tensor, img2: torch.Tensor):
    #Convert range to [-1.0, 1.0] for Diffusers img2img
    img = 2.0 * img - 1.0
    img2 = 2.0 * img2 - 1.0
    return pipe("4K, dark", latents=lat, image = img, image2 = img2, strength=0.45, num_inference_steps = 50, img1_strength=0.7, output_type="np.array").images[0]


#Convert to FP16 RGB
src = clip
clip = core.resize.Spline36(clip, format = vs.RGBH, matrix_in_s="709", width=512, height=512)
forward_comp = core.resize.Spline36(forward_comp, format = vs.RGBH, matrix_in_s="709", width=512, height=512)

#run inference
processed = utils.inference.videoinference(clip, forward_comp, diffuse)

#clip = core.std.MakeDiff(processed, clip)

processed.set_output()