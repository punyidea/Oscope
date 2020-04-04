import os

#os.environ['PATH'] += ":/usr/local/Cellar/ffmpeg/4.2.2_1/bin:/opt/local/bin:/opt/local/var:/opt/local/sbin:/usr/local/sbin:/Library/Frameworks/Python.framework/Versions/3.5/bin"
from pydub import AudioSegment
from pydub.utils import which
import numpy as np
import sys
import warnings



def write_file(fname,aud_segment,np_array,def_ext='mp4',fs_out = 192000):
    '''

    :param fname:
    :param aud_segment:
    :param np_array: of dim, N,num_channels, ALREADY in final state
    :param dtype: the original datatype associated to the audio segment
    :return:
    '''
    assert(np_array.ndim==1 or np.size(np_array,1)==aud_segment.channels )
    if np_array.dtype != determine_dtype(aud_segment):
        warnings.warn('The data type of the array is not as expected by the program.')
        np_array = np.asarray(np_array,dtype=determine_dtype(aud_segment))
    aud_segment.__setattr__('_data', np_array.tobytes('C'))
    aud_segment= aud_segment.set_frame_rate(fs_out)

    f_root,f_format = os.path.splitext(fname)[-2:]
    f_format = f_format[1:]
    AudioSegment.converter = which("ffmpeg")
    #np_array.tofile('{}.{}'.format(fname,'bin'))
    aud_segment.export(fname, format=(f_format if f_format else def_ext),#kwargs={'-vcodec':'codec','-acodec':'codec'}
                        )

def convert_float_dtype(np_array,aud_segment):
    if np.max(np.abs(np_array))>1:
        Warning('Signal is scaled such that values will clip.')
    dtype_seg = determine_dtype(aud_segment)
    max_scale= -np.iinfo(dtype_seg).min
    converted_array = np.asarray(np_array*max_scale,dtype = dtype_seg)
    return  converted_array

def determine_dtype(aud_segment):
    '''
    To be expanded if the format for float is seen.
    A bit hacky.'''
    dtype_dict = {'i':'int'}
    prefix = dtype_dict[aud_segment.array_type]
    dtype_str  ='{}{}'.format(prefix,aud_segment.sample_width*8)
    return np.dtype(dtype_str)