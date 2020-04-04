import unittest
import numpy as np
import numpy.testing
import array

import copy
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from Audio.circle_transform import circle_transform
from Audio.sound_io import convert_float_dtype,write_file
from Audio.preprocess import normalize
import os
#os.environ['PATH'] += ":/usr/local/Cellar/ffmpeg/4.2.2_1/bin:"
print(os.environ['PATH'])
from pydub import AudioSegment
from Graphics.Draw import poly_H,poly_E,poly_L,\
    path_H,path_L,path_E,\
    render_path_time,render_path_once,render_interpolate_time
from Graphics.Paths import RegPolygon,Polygon,Path,make_sierpinski_triangle



test_file = '/Users/vsp/Music/iTunes/iTunes Media/Music/Kimya Dawson/Remember That I Love You/07 I Like Giants.m4a'
test_out = 'animated_hell_path.mp4'




class TestRead(unittest.TestCase):
    fs = 44100

    aud_seg = AudioSegment(data=[],frame_rate=fs,channels=2,sample_width=4)

    def test_fileio(self):
        aud_seg = AudioSegment.from_file(test_file)

        first_5_seconds = aud_seg[:50000]
        fs = aud_seg.frame_rate
        data_to_process = np.array(first_5_seconds.get_array_of_samples()).reshape([-1,2])
        max_val,orig_dtype = np.abs(data_to_process).max(),data_to_process.dtype
        data_to_process= np.asarray(data_to_process,dtype='float64')/max_val

        [X,Y]=circle_transform(data_to_process,fs)
        new_data = convert_float_dtype(np.stack([X,Y],axis=1,),aud_seg)
        #write_file(test_out,aud_seg,new_data)
        self.assertTrue(True)


    def test_setting_hello_world(self):
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

        animated_data = convert_float_dtype(full_data,first_5_seconds,max_val)
        write_file(test_out,first_5_seconds,animated_data)
        self.assertTrue(True)

    def test_mult_triangle(self):
        test_out = 'triangle_anim.wav'
        f_triangle = 150 #100 Hz
        t_per_draw = 1/f_triangle
        t_per_letter = 5 #5 s per triangle


        fs = self.fs

        triangle_coords = np.array([[0,1],[-1,-1],[0,-1],[1,-1]])
        norm_triangle = Polygon(triangle_coords, loop=True)
        scaled_triangle = copy.deepcopy(norm_triangle)


        coords_norm = render_path_time(norm_triangle, t_per_letter, t_per_draw, fs)

        t_transition = 10
        scale_fact_vect = np.linspace(3, 50, f_triangle * t_transition)
        coords_scaled = render_path_time(scaled_triangle, t_per_draw, t_per_draw, fs)
        n_samples = coords_scaled.shape[0]
        transition_1_coords = np.zeros((n_samples*t_transition*f_triangle,coords_scaled.shape[1]))

        for ind,scale_fact in enumerate(scale_fact_vect):
            scaled_triangle.reparameterize_t([0,1/scale_fact,.5,1-1/scale_fact])
            #coords_norm_scaled = render_interpolate_time(norm_triangle, scaled_triangle, t_per_letter, t_per_draw, fs)
            transition_1_coords[ind * n_samples:(ind + 1) * n_samples, :] \
                = render_path_time(scaled_triangle, t_per_draw, t_per_draw, fs)

        coords_scaled_norm = render_interpolate_time(scaled_triangle, norm_triangle, t_per_letter, t_per_draw, fs)


        full_data = normalize(np.concatenate([ coords_norm,
                                     transition_1_coords,
                                     #transition_2_coords,
                                     coords_scaled_norm
                                   ],axis=0))*.95

        animated_data = convert_float_dtype(full_data, self.aud_seg)
        write_file(test_out,self.aud_seg,animated_data)
        self.assertTrue(True)

    def test_sierpinski(self):
        test_out = 'sierpinski_triangle_path.wav'
        f_triangle = 450  # 100 Hz
        t_per_draw = 1 / f_triangle
        t_per_letter = 3  # 5 s per triangle


        fs = self.fs


        norm_triangle = Path(RegPolygon(3,1).coords,loop=True)
        triangle_list = []
        t_per_draw_layer = t_per_draw
        for i in range(5):
            sier_triangle = make_sierpinski_triangle(i,1)
            level_coords = render_path_time(sier_triangle,t_per_letter,t_per_draw_layer,fs)
            t_per_draw_layer *= np.sqrt(3)
            triangle_list.append(level_coords)

        interp_coords = render_interpolate_time(sier_triangle,norm_triangle,t_per_letter,t_per_draw*3,fs)
        triangle_list.append(interp_coords)

        full_data = np.concatenate(triangle_list, axis=0)
        animated_data = convert_float_dtype(normalize(full_data), self.aud_seg)
        write_file(test_out, self.aud_seg, animated_data)
        self.assertTrue(True)