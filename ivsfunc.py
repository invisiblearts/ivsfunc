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


# The preview is done in a ipython/jupyter environment inline; thus IPython is required.
def preview_frame(src, frame_num, **kwargs):
    from IPython import get_ipython
    from mvsfunc import Preview
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    if src.format.color_family is not vs.GRAY:
        src = Preview(src, **kwargs)
    
    arr = get_array(src, frame_num)
    plt.imshow(arr)
