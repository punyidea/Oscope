import numpy as np
import scipy as sp
import scipy.interpolate as sp_interp

class Path(object):
    '''
    k_interp is the k used in spline interpolation. 3 in default path, and 1 for a polygon.
    '''
    k_interp=3

    def __init__(self,coords):
        '''

        :param coords: A list of coordinates of the object. Shape [npoints,ndims]
        :return tot_len: the total length of the path (euclidean distance between collective points)
        :return self.t: a special list of the t values each point has. t \in [0,1] and
            is such that a constant distance is travelled for each t.
        :return spline_mod: The underlying spline model of the path.
        '''
        self.coords = coords
        t_raw = self.get_cum_path_len(coords)
        self.tot_len,self.t = t_raw[-1], t_raw/t_raw[-1]
        self.spline_mod,_ = sp_interp.splprep(coords.T,u=self.t,k=self.k_interp,s=0)

    def update_coords(self,coords):
        t_raw = self.get_cum_path_len(coords)
        self.tot_len,self.t = t_raw[-1], t_raw/t_raw[-1]
        self.spline_mod,self.t = sp_interp.splprep(coords.T,k=self.k_interp,s=0)

    @staticmethod
    def calc_path_lengths(coords):
        diff_var = np.diff(coords,axis=0)
        return np.sqrt(np.einsum('ij,ij->i',diff_var,diff_var))

    @staticmethod
    def get_cum_path_len(coords):
        seg_lens = Path.calc_path_lengths(coords)
        t_raw = np.cumsum(np.concatenate(([0],seg_lens)))
        return t_raw


    def eval_coords(self,des_t):
        return np.array(sp_interp.splev(des_t,self.spline_mod))

class Polygon(Path):
    k_interp = 1