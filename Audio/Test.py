import unittest
import numpy as np
import numpy.testing
import array

from Audio.circle_transform import circle_transform
from Audio.io import convert_dtype,write_file
import os
os.environ['PATH'] += ":/usr/local/Cellar/ffmpeg/4.2.2_1/bin:"+\
"/Library/Frameworks/Python.framework/Versions/3.5/bin:/opt/local/bin:/opt/local/sbin:/Library/Frameworks/Python.framework/Versions/2.7/bin:/usr/local/sbin:/Library/Frameworks/Python.framework/Versions/3.4/bin:/opt/local/bin:/opt/local/sbin:/Library/Frameworks/Python.framework/Versions/3.4/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/Cellar:"
from pydub import AudioSegment
from pydub.utils import audioop
from Graphics.Draw import poly_H,poly_E,poly_L,\
    render_path_time,render_path_once,render_interpolate_time



test_file = '/Users/vsp/Music/iTunes/iTunes Media/Music/Kimya Dawson/Remember That I Love You/07 I Like Giants.m4a'
test_out = 'five_secs.mp4'

class TestRead(unittest.TestCase):
    def test_fileio(self):
        aud_seg = AudioSegment.from_file(test_file)

        first_5_seconds = aud_seg[:50000]
        fs = aud_seg.frame_rate
        data_to_process = np.array(first_5_seconds.get_array_of_samples()).reshape([-1,2])
        max_val,orig_dtype = np.abs(data_to_process).max(),data_to_process.dtype
        data_to_process= np.asarray(data_to_process,dtype='float64')/max_val

        [X,Y]=circle_transform(data_to_process,fs)
        new_data = convert_dtype(np.stack([X,Y],axis=1,),aud_seg,max_val)
        #write_file(test_out,aud_seg,new_data)
        self.assertTrue(True)


    def test_setting(self):
        aud_seg = AudioSegment.from_file(test_file)

        first_5_seconds = aud_seg[:50000]
        fs = aud_seg.frame_rate
        data_to_process = np.array(first_5_seconds.get_array_of_samples()).reshape([-1, 2])
        max_val, orig_dtype = np.abs(data_to_process).max(), data_to_process.dtype
        data_to_process = np.asarray(data_to_process, dtype='float64') / max_val

        [X, Y] = circle_transform(data_to_process, fs)

        t_per_draw = .1
        t_per_letter = 5
        coords_h = render_path_time(poly_H,t_per_letter,t_per_draw,fs)
        coords_h_e = render_interpolate_time(poly_H,poly_E,t_per_letter,t_per_draw,fs)
        coords_e = render_path_time(poly_E,t_per_letter,t_per_draw,fs)
        coords_e_l = render_interpolate_time(poly_E,poly_L,t_per_letter,t_per_draw,fs)
        coords_l =render_path_time(poly_L,t_per_letter,t_per_draw,fs)

        coords_letters = np.concatenate((coords_h,
                                   coords_h_e,
                                   coords_e,
                                   coords_e_l,
                                   coords_l), axis=1).T

        music_fact = 0.05
        full_data = music_fact*np.stack([X,Y],axis=1)
        full_data[:np.shape(coords_letters)[0]] +=coords_letters
        full_data /= np.max(full_data)

        animated_data = convert_dtype(full_data,first_5_seconds,max_val)


        test_out = 'animated_hell.wav'
        write_file(test_out,first_5_seconds,animated_data)
        self.assertTrue(False)