import numpy as np
from Graphics.Paths import Path,Polygon
from Graphics.MultiPath import MultiPath

coords_H = np.array([[-1, -1],
                     [-1, 1],
                     [-1. / 3, 1],
                     [-1. / 3, 1. / 3],
                     [1. / 3, 1. / 3],
                     [1. / 3, 1],
                     [1, 1],
                     [1, -1],
                     [1. / 3, -1],
                     [1. / 3, -1. / 3],
                     [-1. / 3, -1. / 3],
                     [-1. / 3, -1],
                     ])
poly_H = Polygon(coords_H,loop=True)
path_H = Path(coords_H,loop=True)
coords_E = np.array([[-1, -1],
                     [-1, 1],
                     [1, 1],
                     [1, 3. / 5],
                     [-1. / 3, 3 / 5],
                     [-1. / 3, 1 / 5],
                     [1. / 3, 1. / 5],
                     [1. / 3, -1. / 5],
                     [-1. / 3, -1. / 5],
                     [-1. / 3, -3. / 5],
                     [1, -3. / 5],
                     [1, -1],
                     ])
poly_E = Polygon(coords_E,loop=True)
path_E = Path(coords_E,loop=True)

coords_L = np.array([[-1, -1],
                     [-1, 1],
                     [-1 / 3, 1],
                     [-1/ 3, -1 / 3],
                     [1, -1 / 3],
                     [1, -1],
                     ]
                    )
poly_L = Polygon(coords_L,loop=True)
path_L = Path(coords_L,loop=True)



def interp_paths(p_0,p_1,interp_t_range, des_t = np.linspace(0,1)):
    '''
    Interpolates paths between two polygons.
    :param des_t: The path_rot will look like p_0, at t <= interp_t_range[0] and p_1 at t >= interp_t_range[1]
    :param interp_t_range: range during which we do interpolation.
    :param p_0: parameterized path_rot at the start, with assumed parameterization t \in [0,1]
    :param p_1: parameterized path_rot at the end
    :return: coordinates along the path_rot.
    '''

    t_alpha =np.clip((des_t-interp_t_range[0])/np.diff(interp_t_range),0,1)[:,None]
    p_0_coords = p_0.eval_coords(np.mod(des_t,1))
    p_1_coords = p_1.eval_coords(np.mod(des_t,1))

    return (1-t_alpha)*p_0_coords + t_alpha*p_1_coords

def render_path_once(path_to_draw, t_on_screen, fs):
    '''Draws the path_rot indicated in path_to_draw.'''
    path_coords = render_path_time(path_to_draw, t_on_screen, t_per_draw=t_on_screen,fs=fs)
    return path_coords
def render_path_time(path_to_draw,t,t_per_draw,fs):
    '''
    Loops around the path_rot if t > t_per_draw
    :param path_to_draw: parameterized path_rot at the start, with assumed parameterization t \in [0,1]
    :param t: total time elapsed while drawing (s)
    :param t_per_draw: Time elapsed in a single rendering of the path_rot (s)
    :param fs: sampling of drawing (Hz)
    :return: path_shape, t*fs = num samples
    '''
    t_path = np.mod(np.arange(float(t)*float(fs))/(fs*t_per_draw),1)
    path_coords= path_to_draw.eval_coords(t_path)
    return path_coords

def render_interpolate_time(p_0,p_1, t, t_per_draw, fs):
    '''
    For simplicity assumes that t is the whole time between p_0 and p_1,
    that is, time less than it is not considered.
    :param p_0: parameterized path_rot at the start, with assumed parameterization t \in [0,1]
    :param p_1: parameterized path_rot at the start, with assumed parameterization t \in [0,1]
    :param t: total time elapsed while drawing (s)
    :param t_per_draw: Time elapsed in a single rendering of the paths (s)
    :param fs: sampling of drawing (Hz)
    :return: path_shape,t*fs
    '''
    interp_t_range = np.array([0,(t*fs -1)/(fs*t_per_draw)])
    t_path = np.arange(t * fs) / (fs* t_per_draw)
    path_coords = interp_paths(p_0,p_1,interp_t_range,t_path)
    return path_coords

