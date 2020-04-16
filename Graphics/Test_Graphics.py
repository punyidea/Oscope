import unittest
import numpy as np
import numpy.testing as np_tests
import matplotlib.pyplot as plt

from Graphics.Paths import  SplinePath,Polygon,RegPolygon
from Graphics.MultiPath import MultiPath

from Graphics.utils import rot_points2d
from Graphics.fractals import make_sierpinski_triangle_multipath

from Graphics.Draw import render_path_once, interp_paths, \
    poly_H, path_H, poly_E, path_E

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class TestPolygon(unittest.TestCase):
    coords_int_path = np.array([[0, 0], [0, 1], [4, 4], [4, 0], [0, 0]])
    coords_square = np.array([[0,0],[0,1],[1,1],[1,0]])
    coords_triangle = np.array([[0,0],[0,2],[1,2]])
    coords_rect = np.array([[0,0],[0,1],[2,1],[2,0]])
    def test_path_lens(self):
        path_lens = SplinePath.calc_path_lengths(self.coords_int_path)
        np_tests.assert_allclose(path_lens,np.array([1.,5,4,4]))
        self.assertTrue(self,True)

    def test_cum_path_len(self):
        raw_t = SplinePath.get_cum_path_len(self.coords_int_path)
        np_tests.assert_allclose(raw_t,np.array([0,1.,6,10,14]))
        self.assertTrue(self,True)

    def test_Polygon(self):
        def poly_check_statements(poly):
            coord_at_half = poly.eval_coords(.5)
            self.assertAlmostEqual(coord_at_half[0], 4)
            self.assertLess(coord_at_half[1], 4)
            self.assertGreater(coord_at_half[1], 0)

        poly = Polygon(self.coords_int_path)
        np_tests.assert_allclose(poly.t,np.array([0,1.,6,10,14])/14.)
        poly_check_statements(poly)
        SplinePath.validate_path(poly)

        poly_loop = Polygon(self.coords_int_path, loop=True)
        np_tests.assert_allclose(poly_loop.t, np.array([0, 1., 6, 10]) / 14.)
        poly_check_statements(poly_loop)
        SplinePath.validate_path(poly_loop)

    def test_reparameterize(self):
        poly = Polygon(self.coords_int_path,loop=True)
        poly.reparameterize_t([0,.25,.5,.75])

        coord_at_quarter = poly.eval_coords(.25)
        np_tests.assert_allclose(coord_at_quarter, self.coords_int_path[1])
        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half, self.coords_int_path[2])

    def test_set_coords(self):
        poly= Polygon(self.coords_square,loop=True)
        poly.reset_coords(self.coords_rect, reparam=False)
        coord_at_quarter = poly.eval_coords(.25)
        np_tests.assert_allclose(coord_at_quarter, self.coords_rect[1])
        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half, self.coords_rect[2])

        poly.reset_coords(self.coords_rect, reparam= True)
        coord_at_quarter = poly.eval_coords(1./6)
        np_tests.assert_allclose(coord_at_quarter, self.coords_rect[1])
        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half, self.coords_rect[2])

    def test_add_const(self):
        poly = Polygon(self.coords_int_path) + 1
        np_tests.assert_allclose(poly.t, np.array([0, 1., 6, 10, 14]) / 14.)
        np_tests.assert_allclose(poly.center,[1,1])

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords, self.coords_int_path + 1)
        self.assertAlmostEqual(coord_at_half[0],5)

        poly = Polygon(self.coords_int_path,loop=True)
        poly +=[0,1]
        np_tests.assert_allclose(poly.center, [0, 1])

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords, self.coords_int_path[:-1] + [0,1])
        self.assertAlmostEqual(coord_at_half[0], 4)


    def test_sub_const(self):
        poly = Polygon(self.coords_int_path) - 1
        np_tests.assert_allclose(poly.t, np.array([0, 1., 6, 10, 14]) / 14.)

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords, self.coords_int_path - 1)
        self.assertAlmostEqual(coord_at_half[0], 3)

    def test_rot2d(self):

        coord = np.array((.5,.5),)
        rot_coord=rot_points2d(coord,90,np.array((0,0)))
        np_tests.assert_allclose(rot_coord,(-.5,.5))
        rot_coord=rot_points2d(coord,90,np.array((1,0)))
        np_tests.assert_allclose(rot_coord,(.5,-.5))
        rect = Polygon(self.coords_rect,loop=True)
        vert_rect = SplinePath.rot2d(rect, 90)
        coord_at_half = vert_rect.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half, [-1,2])
        self.assertTrue(True)


    def test_add_exact_coords(self):
        poly_1 = Polygon(self.coords_int_path)
        poly_2 = Polygon(self.coords_square,loop=True)
        poly_sum = Polygon.add_exact_coords(poly_1, poly_2)
        #np_tests.assert_allclose(poly_sum.t, np.sort(np.concatenate((np.array([0, 1., 6, 10, 14]) / 14.,np.arange(1,4)/4))))

        poly_1 = Polygon(self.coords_triangle,loop=True)
        poly_2 = Polygon(self.coords_square,loop=True)
        poly_sum = Polygon.add_exact_coords(poly_1,poly_2)
        coord_at_half = poly_sum.eval_coords(1./3)
        #self.assertAlmostEqual(coord_at_half[0],1/3)

        poly_1 = Polygon(self.coords_square,loop=True)
        poly_2 = Polygon(self.coords_rect,  loop=True)
        poly_sum = Polygon.add_exact_coords(poly_1,poly_2)
        coord_at_half = poly_sum.eval_coords(.5)
        self.assertAlmostEqual(coord_at_half[0],3)
        self.assertAlmostEqual(coord_at_half[1],2)

    def test_add_coords_loops(self):
        round_square = SplinePath(self.coords_square, loop=True)
        round_triangle = SplinePath(self.coords_triangle, loop=True)
        poly_sum = Polygon.add_exact_coords(round_square,round_triangle)

    def test_mult_constant(self):
        round_three_edge = SplinePath(self.coords_square)
        mult_const = (round_three_edge + [1,0])*2

        np_tests.assert_allclose(mult_const.coords,self.coords_square*2 + [1,0])
        self.assertAlmostEqual(mult_const.tot_len,6)
        np_tests.assert_allclose(mult_const.center,[1,0])

        mult_vect = (round_three_edge + [.2,3]) * [1, 2]
        np_tests.assert_allclose(mult_vect.coords, self.coords_square * [1,2] + [.2,3])
        np_tests.assert_allclose(mult_vect.center, [.2, 3])
        self.assertAlmostEqual(mult_const.tot_len, 6)

        mult_vect*= [2,1]
        np_tests.assert_allclose(mult_vect.coords, (self.coords_square*2 + [.2, 3]) )
        np_tests.assert_allclose(mult_vect.center, [.2, 3])

    def test_reg_poly(self):
        triangle = RegPolygon(3, 1)
        square = RegPolygon(4, np.sqrt(2), center=(1, 0))

        square_rot90 = RegPolygon(4, np.sqrt(2), center=(1, 0), ang=90)
        np.testing.assert_allclose(square.coords[np.arange(1, 5) % 4], square_rot90.coords, atol=1e-10)
        np.testing.assert_allclose(square.t, square_rot90.t)
        self.assertAlmostEqual(square.tot_len, square_rot90.tot_len)
        self.assertEqual(square_rot90.k_interp, square.k_interp)

class TestMultiPath(unittest.TestCase):
    coords_shape = np.array([[0, 0], [0, 1], [4, 4], [4, 0], [0, 0]])
    coords_square = np.array([[0,0],[0,1],[1,1],[1,0]])
    path_shape = Polygon(coords_shape)
    path_square = Polygon(coords_square,loop=True)
    def test_init(self):
        x = MultiPath([self.path_shape, self.path_square], None)
        y = MultiPath([self.path_shape, self.path_square], [[0, 14. / 18], [14. / 18, 1]])
        np_tests.assert_allclose(x.t_ints,y.t_ints)
        #self.assertTrue(x.path_list == y.path_list)

        z = MultiPath([self.path_shape, self.path_square], [[0, 1]], center=[0, 2])

    def test_scale(self):
        y = MultiPath([self.path_shape, self.path_square], [[-1, 13], [13, 17]], [1, 1])
        y.rescale_t_ints()
        np_tests.assert_allclose(y.t_ints,[[0,14./18],[14./18,1]])

    def test_sort(self):
        y = MultiPath([self.path_shape, self.path_square], [[13., 17], [-1, 13.]], [1, 1])
        y.sort_paths()
        np_tests.assert_allclose(y.t_ints,[[-1.,13],[13.,17]])

    def test_add_scalar(self):
        x = MultiPath([self.path_shape, self.path_square], None)
        y = x +1
        np_tests.assert_allclose(y.path_list[0].coords,self.coords_shape+1)
        np_tests.assert_allclose(y.center,[1,1])

        y = x +[1,2]
        np_tests.assert_allclose(y.path_list[0].coords,self.coords_shape+[1,2])
        np_tests.assert_allclose(y.center,[1,2])

    def test_add_multpaths(self):
        x = MultiPath([self.path_shape], [0, 1], center=[0,1])
        y = MultiPath([self.path_square],[0,1], center=[1,0])
        z = x + y

        np_tests.assert_allclose(z.t_ints,[[0,1],[0,1]])
        np_tests.assert_allclose(z.center,[1,1])

        coord_at_half_x = x.eval_coords(.5)
        coord_at_half_y = y.eval_coords(.5)
        coord_at_half_z = z.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half_x + coord_at_half_y,coord_at_half_z)
        np_tests.assert_allclose(coord_at_half_x,self.path_shape.eval_coords(.5))
        np_tests.assert_allclose(coord_at_half_y, self.path_square.eval_coords(.5))

    def test_mult_scalar(self):
        x = MultiPath([self.path_square], [0, 1], center=[0,1])
        y = x*2

        np_tests.assert_allclose(y.path_list[0].coords,[[0,-1],[0,1],[2,1],[2,-1]])
        np_tests.assert_allclose(y.t_ints,x.t_ints)
        np_tests.assert_allclose(y.center,x.center)
        self.assertAlmostEqual(y.tot_len,8)
        self.assertAlmostEqual(x.path_list[0].tot_len,4)

        x*=2
        np_tests.assert_allclose(y.path_list[0].coords,x.path_list[0].coords)
        np_tests.assert_allclose(x.t_ints,y.t_ints)
        np_tests.assert_allclose(x.center,y.center)
        self.assertAlmostEqual(x.tot_len,8)

    def test_rot(self):
        x = MultiPath([self.path_shape], [0, 1], center=[0, 1])
        y = MultiPath([self.path_square], [0, 1], center=[0, 1])
        z = x + y

        x_rot = x.rot2d(180,center=[0,0])
        np_tests.assert_allclose(x_rot.path_list[0].coords,-x.path_list[0].coords,atol=1e-10)

        y_rot = y.rot2d(90)
        np_tests.assert_allclose(y_rot.path_list[0].coords,
                                [[1,1],[0,1],[0,2],[1,2]],atol=1e-10)

        x_nested = MultiPath([x],[0,1],center=[0,0])
        x_nested.rot2d_inplace(180)
        np_tests.assert_allclose(x_nested.path_list[0].path_list[0].coords, -x.path_list[0].coords, atol=1e-10)

    def test_eval_coord(self):
        x = MultiPath([self.path_square,self.path_square +1], [[0, .2],[.8,1]], center=[1, 0])

        coords_eval,assigned_ts = x.eval_coords(np.array([.1,.16,.75,.95,1]),ret_assigned_vals=True)
        np_tests.assert_allclose(coords_eval[0], [1,1])
        np_tests.assert_allclose(coords_eval[-2], [2,1])
        np_tests.assert_allclose(coords_eval[-1], [1,1])
        self.assertTrue(np.all(assigned_ts ==[True, True,False,True,True]))

        y = MultiPath([self.path_square, self.path_square + 1], [[0, .5], [.5, 1]], center=[1, 0])

        coords_half= y.eval_coords(.5)
        np_tests.assert_allclose(coords_half, [1, 1])

        #self.assertAlmostEqual(coord_at_half[0], 5)

    def test_flattened(self):
        x = MultiPath([self.path_square,self.path_square +1], [[0, .2],[.8,1]], center=[1, 0])

        y = MultiPath([x,x-1,self.path_square],[[0,.5],[0.5,1],[0.5,1]])

        y.flattened(inplace=True)
        np_tests.assert_allclose(y.t_ints,[[0,.1],[.4,.5],[.5,.6],[.5,1],[.9,1]])

    def test_overlapping_t_ints(self):
        def make_multi_path(t_ints):
            return MultiPath([self.path_square] * t_ints.shape[0], t_ints)
        total_t_ints= np.array([[0,.4]
                     , [.1, .5]
                     , [.4,.9]
                     , [.5,.9]])

        overlapping_ints= [{0,1}, {1,2},{2,3}]

        for ind,subset in enumerate(powerset(range(4))):
            if subset:
                mult_path= make_multi_path(total_t_ints[subset,:])
                expected = any(overlapping_int.issubset(set(subset)) for overlapping_int in overlapping_ints)
                actual = mult_path.check_overlapping_intervals()
                self.assertTrue(expected==actual)


    def test_connected_t_ints(self):
        def make_multi_path(t_ints):
            return MultiPath([self.path_square] * t_ints.shape[0], t_ints)
        total_t_ints= np.array([[0,.4]
                     , [.1, .5]
                     , [.4,.9]
                     , [.5,1.]
                     , [.5,.9]])

        unconnected_subsets = [{0,3},{0,4},{0,3,4}]

        #expected_vec = [None, True, True, True, True, True, True, True, True, False,True, True, True, True, True, True]
                      # 0000, 1000, 0100, 0011, 0100, 0101, 0110, 0111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
        for ind,subset in enumerate(powerset(range(4))):
            if subset:
                mult_path= make_multi_path(total_t_ints[subset,:])
                expected = not (set(subset) in unconnected_subsets)
                actual =mult_path.check_connected_ts()
                self.assertTrue(expected==actual)





class TestDraw(unittest.TestCase):

    @staticmethod
    def plot_render(path_to_draw,n_points=500):
        path_render = render_path_once(path_to_draw,1,n_points).T
        plt.plot(path_render[0],path_render[1])

    def test_plot_path(self):
        self.plot_render(poly_H)
        self.plot_render(SplinePath.add_exact_coords(poly_H, poly_E))
        self.plot_render(SplinePath.add_exact_coords(path_H, path_E))

        self.assertTrue(self,True)

    def test_plot_interpolate(self):
        interp_t_range = np.array([-50000,50000])
        interp_h_e = interp_paths(poly_H,poly_E,interp_t_range,
                     des_t=np.linspace(0.,1.,500))
        plt.plot(interp_h_e[0], interp_h_e[1])
        self.assertTrue(self, True)

    def test_plot_sierpinski(self):
        plot_render = lambda x: plt.plot(x[0],x[1])

        one_iter = make_sierpinski_triangle_multipath(1,1)
        self.plot_render(one_iter)

        plt.clf()
        three_iter = make_sierpinski_triangle_multipath(3,.5,center=[0,.5],ang=90)
        self.plot_render(three_iter)
        a=1


if __name__ == '__main__':
    unittest.main()