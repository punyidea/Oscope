import numpy as np
from Graphics.Paths import SplinePath,RegPolygon,Polygon
from Graphics.MultiPath import MultiPath


def make_sierpinski_triangle(n_iters, radius, center=None, ang=0):
    '''
    Generate a Sierpinski Triangle with n_iters iterations. Deprecated.
    Has radius given by radius and center by center.
    :param n_iters:
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
        ang_extensions = np.arange(3)*2*np.pi/3
        coords_shifts = np.stack((np.cos(ang_extensions),np.sin(ang_extensions)),axis=-1)*\
                        (.5**curr_layer)*radius

        new_pathlist = [[]] * n_triangles
        for ind,path in enumerate(multipath.path_list):
            new_pathlist[ind*3:(ind+1)*3] = [
                path*.5 + coords_shifts[i]
                for i in range(3)]

        return MultiPath(new_pathlist,new_t_ints)
    center = center if center is not None else np.zeros(2)
    start_triangle = RegPolygon(3,radius)
    sier_triangle = MultiPath([start_triangle],[[0,1]])
    for i in range(n_iters):
        sier_triangle = sierpinski_subdivide_path(sier_triangle,radius,i+1)

    return (sier_triangle + center).rot2d_inplace(ang)

def make_sierpinski_triangle_multipath(n_layers,radius,center=None,ang = 0):
    '''
    Generate a Sierpinski Triangle with n_iters iterations.
    Has radius given by radius and center by center.
    :param n_layers:
    :param radius:
    :param center:
    :return:
    '''
    def sierpinski_subdivide_path(multipath, radius,n_sides=3):
        '''
        generates a multipath of the next layer from the present layer.
        :param multipath:
        :param curr_layer:
        :return:
        '''
        ts=np.linspace(0,1,n_sides+1)
        new_t_ints = np.stack((ts[:n_sides],ts[1:]),axis=-1)

        ang_extensions = np.arange(n_sides)*2*np.pi/3
        coords_shifts = np.stack((np.cos(ang_extensions),np.sin(ang_extensions)),axis=-1)*\
                        (.5)*radius

        new_pathlist = [(multipath*.5 + coords_shifts[i]) for i in range(n_sides)]
        return MultiPath(new_pathlist,new_t_ints)

    center = np.array(center) if center is not None else np.zeros(2)
    assert (center.shape==(2,))

    start_triangle = RegPolygon(3,radius)
    sier_triangle = MultiPath([start_triangle],[[0,1]])
    for i in range(n_layers):
        sier_triangle = sierpinski_subdivide_path(sier_triangle,radius,i+1)

    return (sier_triangle.flattened() + center).rot2d_inplace(ang)

def make_van_koch_snowflake(n_iters,radius,center=None,ang=0):
    '''
    radius is the radius of the hexagon (in this case also its side length)
    :param n_iters:
    :param radius:
    :param center:
    :param ang:
    :return:
    '''

    def construct_next_iter_seg(van_koch_line):
        '''
        Assumed initial segment length of radius.
        :param van_koch_line: type(Path,MultiPath)
        :return:
        '''
        scaled_path = van_koch_line*(1./3)
        pathlist = [
                    scaled_path + [1. / 3, 0],
                    (scaled_path + [1. / 12, np.sqrt(3) / 12]).rot2d(-60),
                    (scaled_path + [-1./12,np.sqrt(3)/12]).rot2d(60),
                    scaled_path - [1. / 3, 0],
                    ]

        t_ints = np.stack((np.arange(0,4),np.arange(1,5)),axis=-1)/4
        return MultiPath(pathlist,t_ints,center= van_koch_line.center)

    def make_van_koch_segment(n_iters):
        #make van koch segment of length 1.
        vk_seg= Polygon([[.5,0],[-.5,0]])

        for i in range(n_iters):
            vk_seg = construct_next_iter_seg(vk_seg)
        return vk_seg


    radius = np.array(radius)
    center = center if center is not None else np.zeros(2)
    v_k_seg = make_van_koch_segment(n_iters)

    snowflake_offs_angs = (np.arange(6) + 3/2) * 2*np.pi/6
    seg_offs = np.stack([np.cos(snowflake_offs_angs),np.sin(snowflake_offs_angs)],axis=-1)*np.sqrt(3)/2
    path_list =[(v_k_seg + seg_offs[ind]).rot2d_inplace(ind*60) for ind in range(6)]
    t_ints = np.stack((np.arange(0, 6), np.arange(1, 7)), axis=-1) / 6
    v_k_snowflake = MultiPath(path_list,t_ints)

    return (v_k_snowflake.flattened()*radius + center).rot2d_inplace(ang)


