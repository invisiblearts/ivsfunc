import vapoursynth as vs
import numpy as np

# CHW if not HWC
def get_array(src, frame_num, HWC=True):
    assert(isinstance(src, vs.VideoNode))
    assert(isinstance(frame_num, int))
    assert(isinstance(HWC, bool))
    
    frame = src.get_frame(frame_num)

    out_list = []
    
    for plane in frame.planes:
        out_list.append(np.asarray(plane))

    if HWC:
        return np.dstack(out_list)
    else:
        return np.array(out_list)

def display_array(arr):
    from IPython.display import display, Image
    from io import BytesIO
    from scipy.misc import imsave
    
    f = BytesIO()
    imsave(f, arr, 'png')
    display(Image(f.getvalue()))
    

# The preview is done in a ipython/jupyter environment inline; thus IPython is required.
def preview_frame(src, frame_num, **kwargs):
    from mvsfunc import Preview, Depth

    assert(isinstance(src, vs.VideoNode))
    assert(isinstance(frame_num, int))

    if src.format.color_family is not vs.GRAY:
        src = Preview(src, **kwargs)
    
    arr = get_array(src, frame_num)
    display_array(arr)
