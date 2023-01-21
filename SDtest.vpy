import vapoursynth as vs
import torch

from vapoursynth import core

#Crop and resize input for easy testing
clip = core.ffms2.Source("/home/alpha/Storage/TrainingData/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m")
clip = core.std.Crop(clip, left = 968, right = 968)
#clip = core.resize.Spline36(clip, format = vs.RGBH, matrix_in_s="709", width=512, height=512)
clip = core.resize.Spline36(clip, format = vs.YUV444P16, width=512, height=512)

#Denoise for more temporal consistency
clip = core.knlm.KNLMeansCL(clip, d=2, a=3)

#Generate motion compensated frame.
#TODO: Implement additional compensation with MVTools Flow. Initial testing with it is not generating good results.

depan2 = core.mv.DepanEstimate(clip)
forward_comp = core.mv.DepanCompensate(clip, depan2, offset=1, mirror = 15, subpixel=2, blur = 30)[1:]
clip = clip[:-1] #offset the depan clip by 1 frame so that the "predicted" frames match the clip's frames
#forward_comp = core.text.Text(forward_comp, "forward")
#debugmocomp = core.std.Interleave([clip[:-1], forward_comp])

clip.set_output()