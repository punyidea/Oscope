import numpy as np
import scipy as sp
import scipy.interpolate as sp_interp




class Path(object):
    '''
    k_interp is the k used in spline interpolation. 3 in default path_rot, and 1 for a polygon.
    '''
    k_interp_cls=3

    def __init__(self,coords, t = None, tot_len=None, k_interp =None, loop=False, center = None):
        '''

        :param coords: A list of coordinates of the object. Shape [npoints,ndims]
        :return tot_len: the total length of the path_rot (euclidean distance between collective points)
        :return self.t: a special list of the t values each point has. t \in [0,1] and
            is such that a constant distance is travelled for each t.
        :return spline_mod: The underlying spline model of the path_rot.
        '''
        self.k_interp = k_interp if k_interp is not None else self.k_interp_cls

        self.update_coords(coords,t=t, tot_len=tot_len,loop=loop)
        self.center = np.zeros(self.coords[0].size)

    def update_coords(self,coords,t=None,tot_len=None,loop = False):
        if t is None:
            self.update_coords_from_coords(coords,loop=loop)
        else:
            self.update_coords_manual(coords,t=t,tot_len=tot_len,loop=loop)

    def update_coords_manual(self,coords,t,tot_len,loop):
        if loop:
            Warning('Warning: No support exists yet for manual looping')
        if len(t) != len(coords):
            raise ImportError('Dim. Mismatch. Size t:{}, size coords:{}'.format(t.shape, coords.shape))
        self.spline_mod, self.t = sp_interp.splprep(coords.T, u=t, k=self.k_interp, s=0)
        self.tot_len = tot_len #if tot_len is not None else 1
        self.coords = coords
        self.loop = loop

    def update_coords_from_coords(self,coords,loop=False):
        '''
        Creates a spline interpolation of the given coordinates.
        :param coords:
        :param loop: If true, coords, is assumed to loop back on itself.
        This is important because spline interpolation will use extra information from the last elements.
        :return:
        '''
        self.coords = coords
        self.loop = loop
        t_raw = self.get_cum_path_len(coords)
        self.tot_len,self.t = t_raw[-1], t_raw/t_raw[-1]
        if self.loop :

            last_side_len = np.sqrt(np.dot(coords[0] - coords[-1],coords[0]-coords[-1]))
            if last_side_len ==0:
                self.coords = self.coords[:-1]
                self.t= self.t[:-1]
            n_knots = len(self.coords)
            t = self.t* self.tot_len / (self.tot_len + last_side_len)
            self.tot_len +=last_side_len
            rel_inds = np.arange(-self.k_interp+1,self.k_interp + n_knots)
            t = t[rel_inds % n_knots]
            t[rel_inds<0] -=1; t[rel_inds>=n_knots] +=1
            self.t = t
            self.coords= self.coords[rel_inds% n_knots]
            self.spline_mod,self.t = sp_interp.splprep(self.coords.T,u= self.t,k=self.k_interp,s=0)

    def __add__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path addition is ambiguous. Use one of the supported path_rot addition modules.'))
        else:
            return self._add_const(other)

    def __iadd__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path addition is ambiguous. Use one of the supported path_rot addition modules.'))
        else:
            self.coords += other
            self.center += other
            self.update_coords_manual(self.coords,self.t,self.tot_len,self.loop)

    def _add_const(self, other):
        coords = self.coords + other
        center = self.center + other
        return self.__class__(coords, self.t, self.tot_len,
                              center=center, k_interp=self.k_interp,loop=self.loop)


    def __radd__(self,other):
        return self.__add__(self,other)

    def __mul__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot addition modules.'))
        else:
            return self._mul_const(other)

    def __imul__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot multiplication modules.'))
        else:
            self.coords*= other
            self.center *= other
            self.tot_len *= other
            self.update_coords_manual(self.coords,self.t,self.tot_len,self.loop)


    def _mul_const(self, other):
        coords = self.coords * other
        center = self.center * other
        return self.__class__(coords, self.t, self.tot_len,
                              center=center,
                              k_interp=self.k_interp,loop=self.loop)

    def __rmul__(self,other):
        return self.__mul__(self,other)

    def __neg__(self):
        return -1*self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    @staticmethod
    def rot2d(path_rot, ang):
        '''
        Rotates a path_rot about its center by angle degrees.
        Note: only support 2d for now
        '''

        s,c = np.sin(ang*np.pi/180), np.cos(ang*np.pi/180)
        rot_mat = np.array([[c,-s],[s,c]])
        coords = path_rot.coords - path_rot.center
        new_coords = coords @ rot_mat.T + path_rot.center
        return path_rot.__class__(new_coords, path_rot.t, path_rot.tot_len,
                                  center=path_rot.center, k_interp=path_rot.k_interp,
                                  loop=path_rot.loop)

    @staticmethod
    def translate(path,coords):
        return path + coords

    @staticmethod
    def validate_knots(A):
        if len(A.t)!= len(A.coords):
            raise(ValueError('t and coords are of differing lengths.  '
                             'Size t:{}, size coords:{}'.format(A.t.shape, A.coords.shape)))

    @staticmethod
    def add_exact_coords(A, B):
        '''
        Adds two Paths, based on exact knot coordinates.
        This is achieved by <>. In particular, when two polygons are added,
        the result has the shape obtained when adding for every A(t) + B(t).
        However, the result is then reparameterized so that it passes along the shape with constant arc length.

        Result is found on intersection of t domains.
        Note: For looping shapes, it is absolutely important that parameterization be [0,1].
        :param A,B: Two Path objects, which we will add knot coordinates to make a new Path.
        :return: A new Path object, defined on the intersection of <>.
                In the case of a loop, it's a new loop defined on [0,1].
        '''
        Path.validate_knots(A)
        Path.validate_knots(B)
        if A.loop != B.loop:
            Warning('Warning: Adding a loop to a non-loop. Behavior here is unexpected.')
        loop_res = False
        if A.loop and B.loop:
            loop_res = True
            Warning('Warning: Adding two loops. Logic untested.')
            min_t,max_t = 0,1
            if min_t < min(min(A.t),min(B.t)):
                raise Warning('Extrapolating path_rot values.')
            if max_t> max(max(A.t),max(B.t)):
                raise Warning('Extrapolating path_rot values.')
        else:
            min_t,max_t = max(min(A.t),min(B.t)),min(max(A.t),max(B.t))

        des_inds_A = np.logical_and(A.t >= min_t, A.t <= max_t)
        des_inds_B = np.logical_and(B.t >= min_t, B.t <= max_t)
        A_t,B_t = A.t[des_inds_A],B.t[des_inds_B]

        final_t = np.union1d(A_t,B_t)
        final_t_in_A = np.isin(final_t,A.t)
        final_t_in_B = np.isin(final_t,B.t)

        A_eval_coords = np.stack([A.eval_coords(t) if not final_t_in_A[ind] else np.squeeze(A.coords[np.nonzero(A.t==t)])
                         for ind,t in enumerate(final_t)])
        B_eval_coords = np.stack([B.eval_coords(t) if not final_t_in_B[ind] else np.squeeze(B.coords[np.nonzero(B.t==t)])
                         for ind,t in enumerate(final_t)])
        coords = A_eval_coords + B_eval_coords

        k_interp = max(A.k_interp,B.k_interp)
        return A.__class__(coords = coords,k_interp = k_interp,loop = loop_res)

    @staticmethod
    def calc_path_lengths(coords):
        diff_var = np.diff(coords,axis=0)
        return np.sqrt(np.einsum('ij,ij->i',diff_var,diff_var))


    @staticmethod
    def get_cum_path_len(coords):
        seg_lens = Path.calc_path_lengths(coords)
        t_raw = np.cumsum(np.concatenate(([0],seg_lens)))
        return t_raw

    @staticmethod
    def get_tot_len(coords):
        return sum(Path.calc_path_lengths(coords))

    def eval_coords(self,des_t):
        return np.array(sp_interp.splev(des_t,self.spline_mod))




class MultiPath(Path):
    pass

class Polygon(Path):
    k_interp_cls = 1