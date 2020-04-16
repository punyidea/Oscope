import numpy as np
import numpy.ma as ma
import copy
from Graphics.Paths import SplinePath
from Graphics.utils import  check_listlike, rot_points2d,make_list


class MultiPath(object):
    '''
    An object that stores the concatenation of multiple paths.
    '''
    def __init__(self,path_list,t_ints=None, center = None, tot_len = None,loop=False):
        '''

        :param path_list: length of number of paths used.
        :param t_ints: The sub-interval of the
        For convenience, can be length 1, in which case it is applied to all paths (that is, all paths will be added.
        If this parameter is not set, the default is to concatenate all paths
        and weight them according to their collective path lengths.
        :param scales:
        '''

        (path_list,t_ints,center),ndims = self._valid_inputs(path_list,t_ints,center)
        n_paths = len(path_list)
        if  t_ints is None:
            t_ints = self.t_ints_from_path_lens(path_list)

        #If t_ints was set to be one interval, and there is more than one path, replicate it.
        t_ints = [t_ints[0]] * len(path_list) if n_paths > len(t_ints) == 1  else t_ints

        self.path_list = make_list(copy.deepcopy(path_list))
        self.t_ints = np.array(t_ints)

        self.sort_paths()
        self.center = np.array(center) if center is not None else np.zeros(ndims)
        self.tot_len = tot_len if tot_len is not None else  sum((path.tot_len for path in path_list))
        self.ndims = ndims #number of dimensions of the path.
        self.loop = loop

    def __add__(self,other):
        if issubclass(type(other),MultiPath):
            return self._add_mult_path(other)
        else:
            return self._add_const(other)

    def __iadd__(self, other):
        if issubclass(type(other), MultiPath):
            return self._add_mult_path(other,out=self)
        else:
            return self._add_const(other,out=self)

    def _add_const(self,other,out=None):
        other = np.array(other)
        assert(len(other.shape)<=1)
        res = copy.deepcopy(self) if out is None else out
        for path in res.path_list:
            path._add_const(other,out=path)

        res.center +=  other
        return res

    def _add_mult_path(self,other,out=None):
        assert(self.ndims ==other.ndims)
        res = copy.deepcopy(self) if out is None else out
        res.path_list +=other.path_list
        res.t_ints =np.concatenate((res.t_ints,other.t_ints),axis=0)
        res.tot_len += other.tot_len
        res.center += other.center
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def scale_by_path_center(self,other,out=None):
        '''
        Scales each constituent path in the object by scaling factor other, wrt its center.
        Applies recursively if it's multipath.
        :param other:
        :param out:
        :return:
        '''
        res = copy.deepcopy(self) if out is None else out
        for path in res.path_list:
            path._scale_by_const(other,out=path,reparameterize=True)
        self.update_tot_len()

        return res

    def _scale_by_const(self, other, center = None, out= None):
        '''
        Scales the
        :param other:
        :param center:
            If None, scales by the MultiPath's center.

        :param out:
        :return:
        '''
        center = np.array(center) if center is not None else self.center
        assert(center.shape == (self.ndims,))

        res = copy.deepcopy(self) if out is None else out
        for path in res.path_list:
            path._scale_by_const(other,out=path, center=center)
        res.update_tot_len()

        return res

    def __mul__(self, other):
        if issubclass(type(other), SplinePath):
            raise (NotImplementedError('Path multiplication is ambiguous. '
                                       'Use one of the supported path_rot addition modules.'))
        else:
            return self._scale_by_const(other)

    def __imul__(self, other):
        if issubclass(type(other), SplinePath):
            raise (NotImplementedError('Path multiplication is ambiguous. '
                                       'Use one of the supported path_rot multiplication modules.'))
        else:
            return self._scale_by_const(other,out=self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        other = np.array(other)
        assert(len(other.shape)<=1)
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def update_tot_len(self):
        self.tot_len = sum((path.tot_len for path in self.path_list))

    @staticmethod
    def _valid_inputs(path_list,t_ints,center):

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
            lens = np.fromiter(map(len, [path_list, t_ints if t_ints is not None else [1]]), int,
                               count=2)
        except TypeError:
            raise ValueError(
                'Unexpected type encountered in input. Check path_list,t_ints are iterable.')

        #check path_list
        for ind, path in enumerate(path_list):
            if not issubclass(type(path), (SplinePath, MultiPath)):
                raise ValueError(
                    'Path entry at index {} is not instance of Path or MultiPath. Type:{}'.format(ind, type(path)))
            elif issubclass(type(path), SplinePath):
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
            raise ValueError('Invalid Length assigment. Path_list,t_ints size= {}'.format(lens))

        return (path_list,t_ints,center),ndims

    @property
    def n_paths(self) -> int:
        return self.t_ints.shape[0]

    @staticmethod
    def get_dim_path_list(path_list):
        dims_paths = np.array([path.ndims for path in path_list])
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

    def rot2d_inplace(self,ang,center=None):
        self._rot2d(self,ang,center,out=self)
        return self

    def rot2d(self,ang,center=None):
        #center = self.center if center is None else center
        return self._rot2d(self,ang,center)

    @staticmethod
    def _rot2d(mult_path,ang,center = None,out=None):

        rot_origin =np.array(center) if center is not None else mult_path.center
        assert(rot_origin.shape==(mult_path.ndims,))
        res = out if out is not None else copy.deepcopy(mult_path)

        for path in res.path_list:
            path.rot2d_inplace(ang,center=rot_origin)

        if center is not None:
            new_center = rot_points2d(mult_path.center,ang,rot_origin)
            res.center=new_center
        return res

    @staticmethod
    def _rot2d_about_path_centers(mult_path, ang, out=None):
        '''
        Rotates each path found within the multi
        :param mult_path:
        :param ang:
        :param out:
        :return:
        '''

        res = out if out is not None else copy.deepcopy(mult_path)

        for path in res.path_list:
            if issubclass(type(path),MultiPath):
                path._rot2d_about_path_centers(path,ang,out=path)
            else:
                path.rot2d_inplace(ang)

        return res

    def flattened(self,inplace=False):
        '''
        returns a 'flattened' version of the multipath.
        That is, one in which none of the paths within are multipath objects.
        :return:


        '''
        t_ints = []
        path_list = []

        res =self if inplace is not None else copy.deepcopy(self)

        for ind,path in enumerate(self.path_list):
            if issubclass(type(path),MultiPath):
                mp = path.flattened(inplace=True)
                path_list.extend(mp.path_list)
                min_t,max_t = self.t_ints[ind]
                t_ints.append(mp.t_ints*(max_t-min_t) + min_t)
            else:
                path_list.append(path)
                t_ints.append(np.atleast_2d(self.t_ints[ind,:]))


        t_ints = np.concatenate(t_ints,axis=0)
        res.path_list = path_list
        res.t_ints = t_ints
        res.sort_paths()
        return res

    def to_single_path(self, inplace=False,eps=1e-10,raise_error_unconnected=True):
        def get_field_path(path,field):
            ret_val = getattr(path,field)
            if path.loop:
                ret_val = np.append(ret_val,ret_val[None,0],axis=0)
                if field=='t':
                    ret_val[-1] = 1
            return ret_val

        res = self if inplace is True else copy.deepcopy(self)
        res.flattened(inplace=True)
        if res.check_overlapping_intervals():
            raise NotImplementedError('overlapping paths are not supported at this time.')
        if not res.check_connected_ts():
            warn_str ='Path is not fully connected. Undefined behavior between endpoints.'
            Warning(warn_str)
            if raise_error_unconnected:
                raise NotImplementedError(warn_str)

        coords_list =[np.atleast_2d(get_field_path(path,'coords')) for path in res.path_list]
        #coords_list.append(np.atleast_2d(res.path_list[-1].coords[-1]))


        t_list = [(get_field_path(path,'t')*(t_int[1]-t_int[0]) + t_int[0])
                           for (path,t_int) in zip(res.path_list,res.t_ints)]
        #t_list.append(np.atleast_1d(res.t_ints[-1,1]))
        for i in range(1,res.n_paths):
            if res.t_ints[i,0] == res.t_ints[i-1,1]:
                t_list[i][0] +=eps
                coords_list[i][0] +=eps


        t = np.concatenate(t_list)
        coords = np.concatenate(coords_list)

        k_interp = max(path.k_interp for path in res.path_list)
        if k_interp>1:
            Warning('Undefined behavior for nonlinear interpolation.')

        ret_path = SplinePath(coords,t,tot_len=res.tot_len, center=res.center,
                              loop=self.loop, k_interp=k_interp)
        return ret_path


    def check_overlapping_intervals(self):
        inds = np.argsort(self.t_ints[:, 0])
        #n_paths = self.n_paths
        return any(self.t_ints[inds[1:],0]<self.t_ints[inds[:-1],1])

    def check_connected_ts(self):
        '''
        Checks that the union of t ints is a single connected set.
        :return:
        '''
        def interval_intersects_right(curr_int,check_int):
            return check_int[0] <= curr_int[1] and check_int[1] > curr_int[1]
        def connected_interval_exists_right():
            for j in range(i + 1, n_paths):
                check_int = self.t_ints[inds[j]]
                if check_int[0] > curr_int[1]:
                    break
                if interval_intersects_right(curr_int,check_int):
                    return True

            for j in range(i - 1, -1, -1):
                check_int = self.t_ints[inds[j]]
                if interval_intersects_right(curr_int,check_int):
                    return True

            return False

        def interval_intersects_left(curr_int,check_int):
            return check_int[0] < curr_int[0] and check_int[1] >= curr_int[0]
        def connected_interval_exists_left():
            for j in range(i - 1, -1, -1):
                check_int = self.t_ints[inds[j]]
                if interval_intersects_left(curr_int, check_int):
                    return True
            for j in range(i + 1, n_paths):
                check_int = self.t_ints[inds[j]]
                if check_int[0] > curr_int[0]:
                    break
                if interval_intersects_left(curr_int, check_int):
                    return True
            return False

        inds = np.argsort(self.t_ints[:,0])
        n_paths = self.n_paths
        full_min_t = np.min(self.t_ints)
        full_max_t = np.max(self.t_ints)
        for i in range(n_paths):
            min_t,max_t = curr_int = self.t_ints[inds[i]]

            if max_t < full_max_t:
                if not connected_interval_exists_right():
                    return False

            if min_t > full_min_t:
                if not connected_interval_exists_left():
                    return False
        return True


    def eval_coords(self,des_t,ret_assigned_vals=False,tol=1e-10):
        '''
        Sums along all paths which are "valid" in a interval.
        To support concatenation, the range is min_t (inclusive) to max_t (exclusive)
        :param des_t:
        :param ret_assigned_vals: binary array which is 1 if the value was assigned to.
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
            invalid_cond = np.asarray(np.logical_or.reduce([min_t >des_t, des_t > max_t, global_mask])) # note max bound is strict.
            if not np.all(self.t_ints[ind+1:,:]-max_t> tol) or not np.all(max_t-self.t_ints[max(ind-1,0)::-1] >tol):
                invalid_cond |= (des_t ==max_t)

            if not np.all(invalid_cond):
                des_t_mask = ma.masked_where(invalid_cond,des_t)
                eval_ts = (des_t_mask-min_t)*1/(max_t-min_t)

                if issubclass(type(path),MultiPath):
                    coords,invalid_cond = path.eval_coords(eval_ts,ret_assigned_vals=True)
                else:
                    coords= path.eval_coords(eval_ts)
                eval_coords += coords
                assigned_ts |= ~invalid_cond

        if ret_assigned_vals:
            return eval_coords,assigned_ts
        else:
            return eval_coords

