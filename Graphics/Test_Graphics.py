import unittest
import numpy as np
import numpy.testing as np_tests
import matplotlib.pyplot as plt

from Graphics.Paths import  Path,Polygon, MultiPath
from Graphics.Draw import render_path_once, interp_paths, \
    poly_H, path_H, poly_E, path_E

class TestPolygon(unittest.TestCase):
    coords = np.array([[0, 0], [0, 1], [4, 4], [4, 0], [0, 0]])
    coords_square = np.array([[0,0],[0,1],[1,1],[1,0]])
    coords_triangle = np.array([[0,0],[0,2],[1,2]])
    coords_rect = np.array([[0,0],[0,1],[2,1],[2,0]])
    def test_path_lens(self):
        path_lens = Path.calc_path_lengths(self.coords)
        np_tests.assert_allclose(path_lens,np.array([1.,5,4,4]))
        self.assertTrue(self,True)

    def test_cum_path_len(self):
        raw_t = Path.get_cum_path_len(self.coords)
        np_tests.assert_allclose(raw_t,np.array([0,1.,6,10,14]))
        self.assertTrue(self,True)

    def test_Polygon(self):
        poly = Polygon(self.coords)
        np_tests.assert_allclose(poly.t,np.array([0,1.,6,10,14])/14.)

        coord_at_half = poly.eval_coords(.5)
        self.assertAlmostEqual(coord_at_half[0],4)
        self.assertLess(coord_at_half[1],4)
        self.assertGreater(coord_at_half[1],0)

    def test_add_const(self):
        poly = Polygon(self.coords) + 1
        np_tests.assert_allclose(poly.t, np.array([0, 1., 6, 10, 14]) / 14.)
        np_tests.assert_allclose(poly.center,[1,1])

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords,self.coords + 1)
        self.assertAlmostEqual(coord_at_half[0],5)

        poly_vect = poly - [1,0]
        np_tests.ass

    def test_sub_const(self):
        poly = Polygon(self.coords) - 1
        np_tests.assert_allclose(poly.t, np.array([0, 1., 6, 10, 14]) / 14.)

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords, self.coords - 1)
        self.assertAlmostEqual(coord_at_half[0], 3)

    def test_rot2d(self):
        rect = Polygon(self.coords_rect,loop=True)
        vert_rect = Path.rot2d(rect,90)
        coord_at_half = vert_rect.eval_coords(.5)
        np_tests.assert_allclose(coord_at_half, [-1,2])
        self.assertTrue(True)


    def test_add_exact_coords(self):
        poly_1 = Polygon(self.coords)
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
        round_square = Path(self.coords_square,loop=True)
        round_triangle = Path(self.coords_triangle,loop=True)
        poly_sum = Polygon.add_exact_coords(round_square,round_triangle)

    def test_mult_constant(self):
        round_three_edge = Path(self.coords_square)
        #TODO: Rewrite test for tot_len
        mult_const = (round_three_edge + [1,0])*2

        np_tests.assert_allclose(mult_const.coords,(self.coords_square + [1,0])*2)
        self.assertAlmostEqual(mult_const.tot_len,6)
        np_tests.assert_allclose(mult_const.center,[2,0])

        mult_vect = (round_three_edge + [.2,3]) * [1, 2]
        np_tests.assert_allclose(mult_vect.coords, (self.coords_square + [.2,3]) * [1,2])
        np_tests.assert_allclose(mult_vect.center, [.2, 6])
        self.assertAlmostEqual(mult_const.tot_len, 6)


class TestMultiPath(unittest.TestCase):
    coords_shape = np.array([[0, 0], [0, 1], [4, 4], [4, 0], [0, 0]])
    coords_square = np.array([[0,0],[0,1],[1,1],[1,0]])
    path_shape = Polygon(coords_shape)
    path_square = Polygon(coords_square,loop=True)
    def test_init(self):
        scale = [1]
        x = MultiPath([self.path_shape, self.path_square], None, scale, t_int_from_path_len=True)
        y = MultiPath([self.path_shape, self.path_square], [[0, 14. / 18], [14. / 18, 1]], [1, 1])
        np_tests.assert_allclose(x.t_ints,y.t_ints)
        np_tests.assert_equal(x.scales,y.scales)
        self.assertTrue(x.path_list == y.path_list)

        z = MultiPath([self.path_shape, self.path_square], [[0, 1]], [1], center=[0, 2])

    def test_scale(self):
        y = MultiPath([self.path_shape, self.path_square], [[-1, 13], [13, 17]], [1, 1])
        y.rescale_t_ints()
        np_tests.assert_allclose(y.t_ints,[[0,14./18],[14./18,1]])

    def test_sort(self):
        y = MultiPath([self.path_shape, self.path_square], [[13., 17], [-1, 13.]], [1, 1])
        y.sort_paths()
        np_tests.assert_allclose(y.t_ints,[[-1.,13],[13.,17]])

    def test_add_scalar(self):
        x = MultiPath([self.path_shape, self.path_square], None, [1], t_int_from_path_len=True)
        y = x +1
        np_tests.assert_allclose(y.path_list[0].coords,self.coords_shape+1)
        np_tests.assert_allclose(y.center,[1,1])

        y = x +[1,2]
        np_tests.assert_allclose(y.path_list[0].coords,self.coords_shape+[1,2])
        np_tests.assert_allclose(y.center,[1,2])

    def test_add_mult_paths(self):
        x = MultiPath([self.path_shape], [0, 1], [1], center=[0,1])
        y = MultiPath([self.path_square],[0,1],[1], center=[1,0])
        z = x + y

        np_tests.assert_allclose(z.t_ints,[[0,1],[0,1]])
        np_tests.assert_allclose(z.scales,[1,1])
        np_tests.assert_allclose(z.center,[1,1])

    def test_eval_coord(self):
        x = MultiPath([self.path_square,self.path_square +1], [[0, .2],[.8,1]], [1], center=[1, 0])

        coords_eval = x.eval_coords(np.array([.1,.16,.75,.95]))
        np_tests.assert_allclose(coords_eval[0], [1,1])
        np_tests.assert_allclose(coords_eval[-1], [2,1])
        #self.assertAlmostEqual(coord_at_half[0], 5)


        #np.testing.assert_allclose()
class TestDraw(unittest.TestCase):

    def test_plot_path(self):
        path_H_plot = render_path_once(poly_H,1,500)
        plt.plot(path_H_plot[0],path_H_plot[1])

        path_HE_poly_plot = render_path_once(Path.add_exact_coords(poly_H, poly_E),1,500)
        plt.plot(path_HE_poly_plot[0],path_HE_poly_plot[1])

        path_HE_path_plot = render_path_once(Path.add_exact_coords(path_H, path_E),1,500)
        plt.plot(path_HE_path_plot[0],path_HE_path_plot[1])

        self.assertTrue(self,True)

    def test_plot_interpolate(self):
        interp_t_range = np.array([-50000,50000])
        interp_h_e = interp_paths(poly_H,poly_E,interp_t_range,
                     des_t=np.linspace(0.,1.,500))
        plt.plot(interp_h_e[0], interp_h_e[1])
        self.assertTrue(self, True)


if __name__ == '__main__':
    unittest.main()