import unittest
import numpy as np
import numpy.testing
import array

from scipy.signal import hilbert

from Audio.circle_transform import circle_transform
from Audio.sound_io import convert_dtype,write_file
import os
#os.environ['PATH'] += ":/usr/local/Cellar/ffmpeg/4.2.2_1/bin:"
print(os.environ['PATH'])
from pydub import AudioSegment
from Graphics.Draw import poly_H,poly_E,poly_L,\
    path_H,path_L,path_E,\
    render_path_time,render_path_once,render_interpolate_time



test_file = '/Users/vsp/Music/iTunes/iTunes Media/Music/Kimya Dawson/Remember That I Love You/07 I Like Giants.m4a'
test_out = 'animated_hell_path.mp4'

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
        data_to_process = np.mean(data_to_process,axis=1)

        H = hilbert(data_to_process)
        [X, Y] = np.real(H), np.imag(H)

        t_per_draw = .2
        t_per_letter = 5
        coords_h = render_path_time(path_H,t_per_letter,t_per_draw,fs)
        coords_h_e = render_interpolate_time(path_H,path_E,t_per_letter,t_per_draw,fs)
        coords_e = render_path_time(path_E,t_per_letter,t_per_draw,fs)
        coords_e_l = render_interpolate_time(path_E,path_L,t_per_letter,t_per_draw,fs)
        coords_l =render_path_time(path_L,t_per_letter,t_per_draw,fs)

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
        write_file(test_out,first_5_seconds,animated_data)
        self.assertTrue(True)