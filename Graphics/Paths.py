import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.interpolate as sp_interp
import copy


def make_list(obj):
    try:
        obj = list(obj)
        return obj
    except TypeError:
        return [obj]

def check_listlike(obj):
    try:
        obj = list(obj)
        return True
    except TypeError:
        return False


class Path(object):
    '''
    k_interp is the k used in spline interpolation. 3 in default path_rot, and 1 for a polygon.
    #TODO: add reparameterize function.
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
        self.center = np.array(center) if center is not None  else np.zeros(self.coords[0].size)

    def update_coords(self,coords,t=None,tot_len=None,loop = False):
        if t is None:
            self.update_coords_from_coords(coords,loop=loop)
        else:
            self.update_coords_manual(coords,t=t,tot_len=tot_len,loop=loop)
        self.ndims = self.coords.shape[-1]

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
        '''
        Adds constant value to all coordinates (thru broadcasting).
        Note: A+1 adds 1 to all dimensions of coordinates.
        :param other:
        :return:
        '''
        coords = self.coords + other
        center = self.center + other
        return self.__class__(coords, self.t, self.tot_len,
                              center=center, k_interp=self.k_interp,loop=self.loop)


    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot addition modules.'))
        else:
            return self._mul_const(other)

    def __matmul__(self,other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot addition modules.'))
        else:
            ret_var = self._mul_const(other)
            ret_var.reparametrise()
            return ret_var


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
        '''
        TODO:
            - fix scaling of tot_len
            - option to reparameterize in case of non-uniform scaling? TBD
        :param other:
        :return:
        '''


        coords = self.coords * other
        center = self.center * other
        #total scaling. accurate if just one number.
        other_arr = np.atleast_1d(other)
        scale_fact = np.max(np.abs(other_arr))
        tot_len = self.tot_len*scale_fact

        return self.__class__(coords, self.t, tot_len,
                              center=center,
                              k_interp=self.k_interp,loop=self.loop)

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    @staticmethod
    def rot2d(path_rot, ang, center= None):
        '''
        Rotates a path_rot about its center by angle degrees.
        Default value for center is the Path's center
        Note: only support 2d for now
        '''

        center = center if center is not None else path_rot.center
        s,c = np.sin(ang*np.pi/180), np.cos(ang*np.pi/180)
        rot_mat = np.array([[c,-s],[s,c]])
        coords = path_rot.coords - center
        new_coords = coords @ rot_mat.T + center
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
        if A.tot_len < 0:
            raise ValueError('tot_len is not positive.')


    @staticmethod
    def add_exact_coords(A, B):
        '''
        Adds two Paths, based on exact knot coordinates.
        This is achieved by interspersing the knots according to t. In particular, when two polygons are added,
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

        if A.coords.shape[-1] != B.coords.shape[-1]:
            raise ValueError('Dimension mismatch between add args. A.coords shape: {}. B.coords shape: {}'.format(
                A.coords.shape,B.coords.shape))
        if A.loop != B.loop:
            Warning('Warning: Adding a loop to a non-loop. Behavior here is unexpected.')
        loop_res = False
        if A.loop and B.loop:
            loop_res = True
            Warning('Warning: Adding two loops. Logic untested.')
            min_t,max_t = 0,1
            if min_t < min(min(A.t),min(B.t)):
                Warning('Extrapolating path_rot values.')
            if max_t> max(max(A.t),max(B.t)):
                Warning('Extrapolating path_rot values.')
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
        '''
        Evaluates the path coordinates at t values given by des_t.
        If des_t is a MaskedArray, returns 0 outside of valid range.
        :param des_t:
        :return:
        '''
        if ma.isMaskedArray(des_t):
            res = np.zeros(des_t.shape + (self.ndims,))
            des_t_val = des_t[~des_t.mask]
            res[~des_t.mask,:] = np.array(sp_interp.splev(des_t_val,self.spline_mod)).T
        else:
            res = np.array(sp_interp.splev(des_t,self.spline_mod)).T

        return res


class MultiPath(object):
    '''
    An object that stores the concatenation of multiple paths.
    TODO: Method that:
        - stores multiple Path instances
        - eval_path(t) which is vectorized
        - add overload v
        - concatenate method - same as add overload?
        - rotate method
        - rescale method v
    '''
    def __init__(self,path_list,t_ints, scales, t_int_from_path_len=False, center = None, tot_len = None):
        '''

        :param path_list: length of number of paths used.
        :param t_ints: The sub-interval of the
        For convenience, can be length 1. Also, if t_int_from_path_len=True

        :param scales:
        '''

        (path_list,t_ints,scales,center),ndims = self._valid_inputs(path_list,t_ints,scales,center)
        n_paths = len(path_list)
        if t_int_from_path_len and t_ints is None:
            t_ints = self.t_ints_from_path_lens(path_list)

        scales = make_list(scales)
        t_ints = [t_ints[0]] * len(path_list) if len(t_ints) == 1 else t_ints
        scales = [scales[0]] * len(path_list) if len(scales) == 1 else scales

        self.path_list = make_list(copy.deepcopy(path_list))
        self.scales = np.ones((n_paths,ndims))
        for i in range(n_paths):
            self.scales[i] *= scales[i]
        self.t_ints = np.array(t_ints)
        self.center = np.array(center) if center is not None else np.zeros(ndims)
        self.tot_len = tot_len if tot_len is not None else  sum((path.tot_len for path in path_list))
        self.ndims = ndims #number of dimensions of the path.

    def __add__(self,other):

        if issubclass(type(other),MultiPath):
            if other.ndims != self.ndims:
                raise ValueError('Adding values with inconsistent dimensions. self:{} != other:{}'.format(
                    self.ndims,other.ndims))
            path_list = self.path_list + other.path_list
            t_ints = np.concatenate((self.t_ints, other.t_ints),axis=0)
            scales = np.concatenate((self.scales,other.scales), axis=0)
            tot_len = self.tot_len + other.tot_len
            center =self.center + other.center

        else:
            path_list = [path + other for path in self.path_list]
            t_ints = self.t_ints
            scales = self.scales
            tot_len = self.tot_len
            center = self.center + other

        return self.__class__(path_list, t_ints, scales,
                              tot_len=tot_len, center=center)

    def __iadd__(self, other):
        if issubclass(type(other), MultiPath):
            if other.ndims != self.ndims:
                raise ValueError('Adding values with inconsistent dimensions. self:{} != other:{}'.format(
                    self.ndims,other.ndims))
            self.path_list +=  other.path_list
            self.t_ints += other.t_ints
            self.scales +=  other.scales
            self.tot_len += other.tot_len
            self.center +=  other.center
        else:
            for path,ind in enumerate(self.path_list):
                self.path_list[ind] += other
            self.center +=other

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def _valid_inputs(path_list,t_ints,scales,center):

        #check t_ints, make sure it is a list of lists. If not, make it one.
        if check_listlike(t_ints):
            if not check_listlike(t_ints[0]):
                t_ints = [t_ints]

            t_lens = np.fromiter(map(len, t_ints), int, count=-1)
            for ind,t_int in enumerate(t_ints):
                if len(t_int)!=2:
                    raise ValueError('t__ints lens expected to be of length 2 throughout. Length of each element: {}'.format(t_lens))
                if t_int[1] < t_int[0]:
                    Warning('t_int at index {} had upper bound lower than lower bound.'.format(ind))
        elif t_ints is not None:
                raise ValueError('Unexpected input for t_ints.')

        # check length of elements
        try:
            lens = np.fromiter(map(len, [path_list, t_ints if t_ints is not None else [1], scales]), int,
                               count=3)
        except TypeError:
            raise ValueError(
                'Unexpected type encountered in input. Check path_list,t_ints,scales are iterable.')

        #check path_list
        for ind, path in enumerate(path_list):
            if not issubclass(type(path), Path) or issubclass(type(path), MultiPath):
                raise ValueError(
                    'Path entry at index {} is not instance of Path or MultiPath. Type:{}'.format(ind, type(path)))
            if path.spline_mod[0].min() >0 or path.spline_mod[0].max() <1:
                Warning('Multipath will potentially extrapolate path at index {}'.format(ind))
        ndims = MultiPath.get_dim_path_list(path_list)

        #check center
        if center is not None:
            if len(center) != ndims:
                raise ValueError('Inconsistent dimensions between center and paths. '
                                 'Expected center dims: {}, Actual: {}'.format(ndims,len(center)))

        if not np.all(np.logical_or(lens==1 ,lens == max(lens)))\
                or lens[0] !=max(lens) or max(lens)<1:
            raise ValueError('Invalid Length assigment. Path_list,t_ints,scales size= {}'.format(lens))

        return (path_list,t_ints,scales,center),ndims

    @staticmethod
    def get_dim_path_list(path_list):
        dim_func = lambda x : x.coords.shape[-1]
        dims_paths = np.fromiter(map(dim_func,path_list),int,len(path_list))
        if not np.all(dims_paths==dims_paths[0]):
            raise ValueError('Expected all paths to be of dim {} (from first path). Actual dimensions: {}.'.format(
                        dims_paths[0], dims_paths))
        else:
            return dims_paths[0]

    @staticmethod
    def t_ints_from_path_lens(path_list):
        t_ints = []
        tot_len = sum(path.tot_len for path in path_list)
        dist_travel =0
        for path in path_list:
            t_ints.append(np.array([dist_travel,dist_travel +path.tot_len])/tot_len)
            dist_travel += path.tot_len
        return t_ints

    def rescale_t_ints(self):
        '''
        Rescales the time intervals so that the range of the multi-path is from 0 to 1.

        :return:
        '''
        min_ts = np.min(self.t_ints[:,0])
        max_ts = np.max(self.t_ints[:,1])
        self.t_ints = (self.t_ints - min_ts) *1/(max_ts-min_ts)

    def sort_paths(self):
        '''
        Sorts all of the paths by the start of the t_int of each path.
        :return:
        '''
        inds = np.argsort(self.t_ints[:,0])
        self.path_list = [self.path_list[ind] for ind in inds]
        self.t_ints = self.t_ints[inds]
        self.scales = self.scales[inds]

    def eval_coords(self,des_t,ret_assigned_vals=False):
        '''
        Sums along all paths which are "valid" in a interval.
        To support concatenation, the range is min_t (inclusive) to max_t (exclusive)
        :param des_t:
        :param ret_assigned_vals:
        :return:
        '''
        if ma.isMaskedArray(des_t):
            global_mask = des_t.mask
        else:
            des_t = np.asarray(des_t)
            global_mask = np.zeros_like(des_t,bool)
        assigned_ts = np.zeros(des_t.shape,bool)

        eval_coords= np.zeros(des_t.shape + (self.ndims,))
        for ind, path in enumerate(self.path_list):
            min_t,max_t = self.t_ints[ind]
            invalid_cond = np.logical_or.reduce([min_t >des_t, des_t >= max_t, global_mask]) # note max bound is strict.
            des_t_mask = ma.masked_where(invalid_cond,des_t)
            eval_ts = (des_t_mask-min_t)*1/(max_t-min_t)

            if issubclass(type(path),MultiPath):
                coords,invalid_cond = path.eval_coords(eval_ts,ret_assigned_vals=True)
            else:
                coords= path.eval_coords(eval_ts)
            eval_coords += self.scales[ind]*coords
            assigned_ts |= ~invalid_cond

        if ret_assigned_vals:
            return eval_coords,assigned_ts
        else:
            return eval_coords


    def _mul_const(self, other):
         scales = self.scales * other
         center = self.center * other
         ret_obj = copy.deepcopy(self)

         # total scaling. TODO: fix tot_len logic, implementing function to compute it.
         other_arr = np.atleast_2d(other)
         scale_fact = np.mean(np.sqrt(np.einsum('ij,ij->i', other_arr, other_arr)))
         tot_len = self.tot_len * scale_fact

         ret_obj.tot_len = tot_len
         ret_obj.scales = scales
         ret_obj.center = center
         return ret_obj


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
            self.scales *= other
            self.center *= other

            # total scaling.
            other_arr = np.atleast_2d(other)
            scale_fact = np.abs(np.mean(np.sqrt(np.einsum('ij,ij->i', other_arr, other_arr))))
            self.tot_len *= scale_fact


    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other
class Polygon(Path):
    k_interp_cls = 1