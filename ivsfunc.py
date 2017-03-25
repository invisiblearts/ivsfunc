import vapoursynth as vs
import numpy as np


# CHW if not HWC
def get_array(src, frame_num, HWC=True):
    assert(isinstance(src, vs.VideoNode))
    assert(isinstance(frame_num, int))
    assert(isinstance(HWC, bool))
    

    frame = src.get_frame(frame_num)
    num_planes = frame.format.num_planes

    out_list = []

    for i in range(num_planes):
        arr = np.array(frame.get_read_array(i))
        out_list.append(arr)

    # check to prevent subsampling issues
    shape = arr.shape
    for arr in out_list:
        assert(arr.shape == shape)

    if HWC:
        return np.dstack(out_list)
    else:
        return np.array(out_list)

