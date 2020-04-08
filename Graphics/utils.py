import numpy as np

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

def compute_rotmat(ang):
    '''
    Rotation is in degrees, ccw
    :param ang:
    :return:
    '''
    s, c = np.sin(ang * np.pi / 180), np.cos(ang * np.pi / 180)
    return np.array([[c, -s], [s, c]])

def rot_points2d(points,ang,origin, rot_mat=None):
    '''
    Roates points in 2d space, ccw. coordinates are
    :param points: shape[-1] ==2
    :param ang:
    :param origin: origin about which to rotate.
    :param rot_mat:
    :return:
    '''

    assert (points.shape[-1]==2 and origin.shape==(2,))
    if rot_mat is None:
        rot_mat = compute_rotmat(ang)
    return (points - origin) @ rot_mat.T + origin

