import vapoursynth as vs
import numpy as np
import ctypes

# CHW if not HWC
def get_array(src, frame_num, HWC=True):
    assert(isinstance(src, vs.VideoNode))
    assert(isinstance(frame_num, int))
    assert(isinstance(HWC, bool))
    

    frame = src.get_frame(frame_num)
    num_planes = frame.format.num_planes

    out_list = []
    
    st = frame.format.sample_type
    bps = frame.format.bytes_per_sample

    if st == vs.INTEGER:
        if bps == 1:
            dtype = ctypes.c_uint8
        elif bps == 2:
            dtype = ctypes.c_uint16
        else:
            raise ValueError('Wrong sample type!')
    elif st == vs.FLOAT:
        if bps == 2:
            raise ValueError('Half sample is not natively supported! Please resample it.')
        elif bps == 4:
            dtype = ctypes.c_float
        else:
            raise ValueError('Wrong sample type!')


    for i in range(num_planes):
        arr = np.ctypeslib.as_array((dtype * frame.width * frame.height).from_address(frame.get_read_ptr(i).value))
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
    from IPython.display import display, Image
    from mvsfunc import Preview
    from scipy.misc import imsave
    from io import BytesIO

    assert(isinstance(src, vs.VideoNode))
    assert(isinstance(frame_num, int))

    if src.format.color_family is not vs.GRAY:
        src = Preview(src, **kwargs)
    
    arr = get_array(src, frame_num)

    f = BytesIO()
    imsave(f, arr, 'png')
    display(Image(f.getvalue()))
