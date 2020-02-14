import os


os.environ['PATH'] += ":/usr/local/Cellar/ffmpeg/4.2.2_1/bin:"+\
"/Library/Frameworks/Python.framework/Versions/3.5/bin:/opt/local/bin:/opt/local/sbin:/Library/Frameworks/Python.framework/Versions/2.7/bin:/usr/local/sbin:/Library/Frameworks/Python.framework/Versions/3.4/bin:/opt/local/bin:/opt/local/sbin:/Library/Frameworks/Python.framework/Versions/3.4/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/Cellar:"
from pydub import AudioSegment
from pydub.utils import audioop
import numpy as np
import sys
import warnings


def write_file(fname,aud_segment,np_array,def_ext='mp4'):
    '''
    A very delicate way to write the data to the object as it does not seem to be working.
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

    f_format = os.path.splitext(fname)[-1]
    aud_segment.export(fname, format=(f_format if f_format else def_ext),#kwargs={'-vcodec':'codec','-acodec':'codec'}
                        )

def convert_dtype(np_array,aud_segment,scaling):
    dtype_str = determine_dtype(aud_segment)
    converted_array = np.asarray(np_array*scaling,dtype = dtype_str)
    return  converted_array

def determine_dtype(aud_segment):
    '''
    To be expanded if the format for float is seen.
    A bit hacky.'''
    dtype_dict = {'i':'int'}
    prefix = dtype_dict[aud_segment.array_type]
    dtype_str  ='{}{}'.format(prefix,aud_segment.frame_width*4)
    return dtype_str