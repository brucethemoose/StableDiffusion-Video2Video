import torch
import vapoursynth as vs
import numpy as np

# adapted from https://github.com/HolyWu/vs-realesrgan/blob/master/vsrealesrgan/__init__.py

if vs.__api_version__.api_major < 4:
    raise Exception("tensor-frame conversion requires VS API > 4")



def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0)

def frame_to_numpy(frame: vs.VideoFrame):
    return np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])


def numpy_to_frame(array, frame: vs.VideoFrame) -> vs.VideoFrame:
    #Output is numpy, not a tensor
    #Shape is (x, y, planes), float32
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[:, :, plane])
    return frame

framedict = dict() #Dictionary to store numpy frames.

def videoinference(clip: vs.VideoNode, clip2: vs.VideoNode, function) -> vs.VideoNode:
    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        #Convert the VS frames to torch tensors
        device = "cuda"
        img = frame_to_tensor(f[0]).to(device, memory_format=torch.channels_last, non_blocking=True)
        if f[0].props.get("_SceneChangePrevious"):
            print("Scene change detected for frame: " + str(n))
            img2 = img
        else:
            img2 = frame_to_tensor(f[1]).to(device, memory_format=torch.channels_last, non_blocking=False)
        
        output = function(img, img2)

        #store the processed frame in the framedict, so they can be accessed later
        framedict[n] = output
        
        return numpy_to_frame(output, f[2].copy())

    new_clip = clip.std.BlankClip(keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, clip2, new_clip], inference), clip_src=[clip, clip2, new_clip]
    )

# Outputs previously processed frames so they can be motion compensated for input into the next step
# Vapoursynth normally does not allow this, and it will break when trying to process more than one frame concurrently
# since the current frame depends on the output of the previous frame
# Ideally DepanCompensate should be ported to pytorch instead
def timemachine(clip: vs.VideoNode, depanest: vs.VideoNode, function) -> vs.VideoNode:
    def getprevious(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        if f[0].props.get("_SceneChangePrevious") or (n == 0):
            # Return the current (unprocessed) frame if a scene change is detected
            # The pipeline will simply merge the frame with itself
            print("Scene change detected")
            if n > 0:
                print(str(n))
                framedict.pop(n-1)
            return f[0]
        else:
            # Otherwise return the previous frame from the framedict
            return numpy_to_frame(framedict.pop(n-1), f[1].copy())

    new_clip = clip.std.BlankClip(keep=True, format = vs.RGBH)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], getprevious), clip_src=[clip, new_clip]
    ) 