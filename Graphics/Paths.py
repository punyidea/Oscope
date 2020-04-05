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
    '''
    k_interp_cls=3

    def __init__(self,coords, t = None, tot_len=None, k_interp =None, loop=False, center = None,**kwargs):
        '''

        :param coords: A list of coordinates of the object. Shape [npoints,ndims]
        :return tot_len: the total length of the path_rot (euclidean distance between collective points)
        :return self.t: a special list of the t values each point has. t \in [0,1] and
            is such that a constant distance is travelled for each t.
        :return spline_mod: The underlying spline model of the path_rot.
        '''
        coords,t,center,self.ndims = self.validate_inputs(coords,t,tot_len,loop,center)
        self.k_interp = k_interp if k_interp is not None else self.k_interp_cls
        self.loop = loop
        self.center = center if center is not None else np.zeros(self.ndims)

        if self.loop:
            coords,t = self.correct_loop_coords_t(coords,t)

        self.coords = coords
        self.reparameterize_t(t=t)
        self.tot_len = tot_len if tot_len is not None else self.tot_len

        self.update_spline_coords(**kwargs)

    @staticmethod
    def validate_inputs(coords,t,tot_len,loop,center):
        coords = np.atleast_2d(coords)
        if center is not None:
            center = np.atleast_1d(center)
            if coords.shape[-1:] != center.shape:
                raise ValueError(
                    'Dim Mismatch! Shape coord:{} shape center:{}'.format(
                    coords.shape,center.shape
                ))
        if t is not None:
            t = np.atleast_1d(t)
            if len(coords) != len(t):
                raise ValueError(
                    'Dim Mismatch! Shape coord: {} shape center: {}'.format(
                coords.shape, t.shape
                ))

        if t is not None and tot_len is None:
            raise ValueError('Tot_len is undefined and t is explicitly chosen.')

        if len(coords) < 2:
            raise ValueError('Fewer than two points in coords_int_path.')

        try:
            loop = bool(loop)
        except:
            raise ValueError('Invalid input for \'loop\'.')

        ndims = coords.shape[-1]
        return coords,t,center,ndims

    def reset_coords(self, coords, center=None, reparam = True,update_spline=True):
        '''
        Sets coordinates to coords_int_path.
        Setting default True reparam to False keeps the existing parameterization.
            Will throw error
        :param coords:
        :param reparam:
        :return:
        '''

        assert (coords.shape[-1] ==self.ndims)
        if self.loop:
            coords,t = self.correct_loop_coords_t(coords,self.t)
        self.coords = coords

        ndims = coords.shape[-1]
        if ndims != self.ndims:
            self.ndims = coords.shape[-1]
            if center is None:
                Warning('Resetting center to zero after change in dimensions.')
                center = np.zeros(ndims)
        if center is not None:
            self.center = center

        if reparam:
            self.reparameterize_t(t=None)

        elif len(coords) != len(self.t):
            raise RuntimeError('Coordinates were set to a different length without')

        if update_spline:
            self.update_spline_coords()

    def correct_loop_coords_t(self,coords,t,tol=1e-14):
        last_side_len = np.sqrt(np.dot(coords[0] - coords[-1], coords[0] - coords[-1]))
        if last_side_len < tol:
            coords = coords[:-1]
            t = t[:-1] if t is not None else None
        return coords,t

    def reparameterize_t(self,t=None,update_spline=True):
        '''
        Reparameterizes t coordinates of path.
        If t is None, this is based on path length between coordinates.
        :param t:
        :return:
        '''
        if t is not None:
            assert(len(t)==len(self.coords))
            t = np.atleast_1d(t)
            self.t = t
        else:
            self.t,self.tot_len = self.calc_t_path_len(self.coords)

        if update_spline:
            self.update_spline_coords()

    def calc_t_path_len(self,coords):
        '''
        Creates a knot parameterization of coordinates based on path length.
        :param coords:
        :param loop: If true, coords_int_path, is assumed to loop back on itself.
        This is important because spline interpolation will use extra information from the last elements.
        :return:
        '''
        t_raw = self.get_cum_path_len(coords)
        tot_len = t_raw[-1]
        t= t_raw/tot_len
        if self.loop:
            t,tot_len = self._correct_loop_t_totlen(self.coords, t,tot_len)

        return t,tot_len

    def _correct_loop_t_totlen(self, coords, t,tot_len):
        '''
        Accounts for the last side length that is not included in coordinates.
        Note that coords_int_path and t cannot "loop back on themselves."
        :param coords:
        :param t:
        :return:
        '''
        last_side_len = np.sqrt(np.dot(coords[0] - coords[-1], coords[0] - coords[-1]))
        t = t * tot_len / (tot_len + last_side_len)
        tot_len += last_side_len
        return t,tot_len

    def set_center(self,center):
        center = np.atleast_1d(center)
        assert(center.shape == (self.ndims,))
        self.center= center

    def set_tot_len(self,tot_len):
        assert(tot_len >=0)
        self.tot_len = tot_len


    def update_spline_coords(self,**kwargs):
        '''
        Updates the internal spline coordinates which are passed to the spline interpolation routine.
        :param kwargs:
        :return:
        '''
        s = kwargs['s'] if 's' in kwargs else 0

        coords,t = self.coords,self.t
        if self.loop:
            coords,t = self.gen_internal_coords_t_loop(coords,t,k_interp=self.k_interp)
        self.spline_mod, _ = sp_interp.splprep(coords.T, u=t, k=self.k_interp, s=s,**kwargs,per=self.loop)

    @staticmethod
    def gen_internal_coords_t_loop(coords,t,k_interp):
        '''
        In the case of a loop, prepend and append the last and first coords_int_path respectively.
        Assumes t goes from 0 to 1, and thus that the domain is equiv. to (t % 1)
        :param coords:
        :param t:
        :return:
        '''
        t = np.append(t,1)
        coords = coords[np.arange(len(coords) + 1) % len(coords)]
        return coords,t


    def __add__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path addition is ambiguous. Use one of the supported path_rot addition modules.'))
        else:
            return self._add_const(other)

    def __iadd__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path addition is ambiguous. Use one of the supported path_rot addition modules.'))
        else:
            self._add_const(other,reparameterize=False,out=self)
            return self

    def _add_const(self, other,reparameterize=False,out=None):
        '''
        Adds constant value to all coordinates (thru broadcasting).
        Note: A+1 adds 1 to all dimensions of coordinates.
        :param other:
        :return:
        '''
        other = np.array(other)
        assert(len(other.shape)<=1)
        res = copy.deepcopy(self) if out is None else out
        coords = self.coords + other
        center = self.center + other

        res.reset_coords(coords, center, reparam=reparameterize)
        return res

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self, other):
        '''
        Multiplication by a constant
        automatically reparameterizes the path to work with arc length.
        :param other:
        :return:
        '''
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot addition modules.'))
        else:
            return self._scale_by_const(other, reparameterize=True)

    def __imul__(self, other):
        if issubclass(type(other),Path):
            raise(NotImplementedError('Path multiplication is ambiguous. '
                                      'Use one of the supported path_rot multiplication modules.'))
        else:
            self._scale_by_const(other, reparameterize=True, out=self)
            return self

    def _scale_by_const(self, other, center=None, reparameterize=True, out=None):
        '''
        Scales coordinates about the object's center.
        :param other: a vector, or scalar by which each coordinate is multiplied to obtain the final result.
        :return:
        '''
        res = copy.deepcopy(self) if out is None else out
        center = self.center if center is None else np.array(center)
        assert(center.shape == (res.ndims,))

        other = np.array(other)
        assert(len(other.shape)<=1)

        coords = (self.coords-center) * other + center
        center = self.center

        res.reset_coords(coords, center, reparam=reparameterize)
        res.update_spline_coords()
        return res

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def irot2d(self,ang,center=None):
        '''
        Rotates the object in place.
        :param ang:
        :param center:
        :return:
        '''
        self._rot2d(self,ang,center=center,out=self)
        return self

    def rot2d(self,ang,center=None):
        '''
        returns a copy of the current object, rotated by ang degrees.
        :param ang:
        :param center:
        :return:
        '''
        return self._rot2d(self, ang, center=center, out=None)

    @staticmethod
    def _rot2d(path_rot, ang, center= None,out=None):
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

        res = copy.deepcopy(path_rot) if out is None else out
        res.reset_coords(new_coords, center, reparam=False)
        res.update_spline_coords()
        return res

    @staticmethod
    def translate(path,coords):
        return path + coords

    @staticmethod
    def validate_path(A):
        if len(A.t)!= len(A.coords):
            raise(ValueError('t and coords_int_path are of differing lengths.  '
                             'Size t:{}, size coords_int_path:{}'.format(A.t.shape, A.coords.shape)))
        if A.tot_len < 0:
            raise ValueError('tot_len is not positive.')
        if A.coords.shape[-1]!= A.ndims or A.center.shape != (A.ndims,):
            raise AssertionError('Number of dims is inconsistent across fields. '
                                 'Dims: Coords;center;ndims'.format(
            ))
        assert(A.k_interp >0 and A.k_interp == int(A.k_interp))

    @staticmethod
    def add_exact_coords(A, B, reparameterize = True):
        '''

        Adds two Paths, based on exact knot coordinates.
        This is achieved by interspersing the knots according to t. In particular, when two polygons are added,
        the result has the shape obtained when adding for every A(t) + B(t).
        If reparameterize is true, the result is then reparameterized so that it passes along the shape with constant arc length.
        Result is found on intersection of t domains.
        Note: For looping shapes, it is absolutely important that parameterization be [0,1].
        :param A,B: Two Path objects, which we will add knot coordinates to make a new Path.
        :return: A new Path object, defined on the intersection of their domains.
                In the case of a loop, it's a new loop defined on [0,1].
        '''
        Path.validate_path(A)
        Path.validate_path(B)

        if A.ndims != B.ndims:
            raise ValueError('Dimension mismatch between add args. A.coords_int_path shape: {}. B.coords_int_path shape: {}'.format(
                A.coords.shape,B.coords.shape))
        if A.loop != B.loop:
            Warning('Warning: Adding a loop to a non-loop. Behavior here is unexpected.')
        loop_res = False
        if A.loop and B.loop:
            #loop_res = True
            # Warning('Warning: Adding two loops. Logic untested.')
            #min_t,max_t = 0,1
            min_t, max_t = min(min(A.t), min(B.t)), max(max(A.t), max(B.t))
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
        center = A.center + B.center

        k_interp = max(A.k_interp,B.k_interp)

        res = copy.deepcopy(A)
        res.reset_coords(coords, center, reparam=True)
        if not reparameterize:
            res.reparameterize(final_t)

        res.update_spline_coords()
        return res

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
    TODO: Redo operator logic, to be cleaner.
    '''
    def __init__(self,path_list,t_ints=None, center = None, tot_len = None):
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
        self.center = np.array(center) if center is not None else np.zeros(ndims)
        self.tot_len = tot_len if tot_len is not None else  sum((path.tot_len for path in path_list))
        self.ndims = ndims #number of dimensions of the path.

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
        if issubclass(type(other), Path):
            raise (NotImplementedError('Path multiplication is ambiguous. '
                                       'Use one of the supported path_rot addition modules.'))
        else:
            return self._scale_by_const(other)

    def __imul__(self, other):
        if issubclass(type(other), Path):
            raise (NotImplementedError('Path multiplication is ambiguous. '
                                       'Use one of the supported path_rot multiplication modules.'))
        else:
            return self._scale_by_const(other,out=None)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
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
            raise ValueError('Invalid Length assigment. Path_list,t_ints size= {}'.format(lens))

        return (path_list,t_ints,center),ndims

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

    def eval_coords(self,des_t,ret_assigned_vals=False):
        '''
        #TODO: add logic for endpoint. Should we support loops too?
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
            eval_coords += coords
            assigned_ts |= ~invalid_cond

        if ret_assigned_vals:
            return eval_coords,assigned_ts
        else:
            return eval_coords

#TODO: Move Polygon up to the top.
class Polygon(Path):
    k_interp_cls = 1


class RegPolygon(Polygon):
    '''
    Convenience class for making regular polygons (in 2d).
    Give number of sides, radius, and center.

    Polygon is assumed to be drawn as endpoints of n_sides equiangular arcs
        of a circle with center at center and radius radius.
    Ang is the angle of CCW rotation, in degrees,
        from having starting coordinate in direction of positive x-axis.
    Center is a length 2 vector describing the center of the polygon in 2d space.
        if center is not given, it is assumed to be centered at the origin.

    '''
    def __init__(self, n_sides, radius, center=None, ang=0, **kwargs):

        if n_sides < 3:
            raise ValueError('Polygon with fewer than 3 sides chosen.')
        if radius < 0:
            raise ValueError('Polygon with negative radius not possible.')
        if center is not None:
            center = np.atleast_1d(center)
            if np.shape(center) != (2,):
                raise ValueError('Center of 2d polygon has unexpected shape.')
        else:
            center = np.zeros(2)

        ang_offset = ang*np.pi/180
        angs = np.arange(n_sides) *np.pi*2/n_sides+ ang_offset
        coords = radius*np.stack((np.cos(angs),np.sin(angs)),axis=-1) + center
        super().__init__(coords, center=center,loop=True,**kwargs)


def make_sierpinski_triangle(n_layers,radius,center=None):
    '''
    Generate a Sierpinski Triangle with n_layers iterations.
    Has radius given by radius and center by center.
    :param n_layers:
    :param radius:
    :param center:
    :return:
    '''
    def sierpinski_subdivide_path(multipath, radius, curr_layer):
        '''
        generates a multipath of the next layer from the present layer.
        :param multipath:
        :param curr_layer:
        :return:
        '''
        n_triangles = 3**curr_layer

        ts=np.linspace(0,1,n_triangles+1)
        new_t_ints = np.stack((ts[:n_triangles],ts[1:]),axis=-1)
        new_scales = np.ones(n_triangles)
        ang_extensions = np.arange(3)*2*np.pi/3
        coords_shifts = np.stack((np.cos(ang_extensions),np.sin(ang_extensions)),axis=-1)*\
                        (.5**curr_layer)*radius

        new_pathlist = [[]] * n_triangles
        for ind,path in enumerate(multipath.path_list):
            new_pathlist[ind*3:(ind+1)*3] = [
                path*.5 + coords_shifts[i]
                for i in range(3)]

        return MultiPath(new_pathlist,new_t_ints,new_scales)


    center = center if center is not None else np.zeros(2)

    start_triangle = Path(RegPolygon(3,radius).coords,loop=True)
    sier_triangle = MultiPath([start_triangle],[[0,1]],[1],)
    for i in range(n_layers):
        sier_triangle = sierpinski_subdivide_path(sier_triangle,radius,i+1)

    return sier_triangle + center