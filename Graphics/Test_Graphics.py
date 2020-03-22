import unittest
import numpy as np
import numpy.testing as np_tests
import matplotlib.pyplot as plt

from Graphics.Paths import  Path,Polygon
from Graphics.Draw import render_path_once, interp_paths, \
    poly_H, path_H, poly_E, path_E

class TestPolygon(unittest.TestCase):
    coords = np.array([[0, 0], [0, 1], [4, 4], [4, 0], [0, 0]])
    coords_square = np.array([[0,0],[0,1],[1,1],[1,0]])
    coords_triangle = np.array([[0,0],[0,2],[1,2]])
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

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords,self.coords + 1)
        self.assertAlmostEqual(coord_at_half[0],5)

    def test_sub_const(self):
        poly = Polygon(self.coords) - 1
        np_tests.assert_allclose(poly.t, np.array([0, 1., 6, 10, 14]) / 14.)

        coord_at_half = poly.eval_coords(.5)
        np_tests.assert_allclose(poly.coords, self.coords - 1)
        self.assertAlmostEqual(coord_at_half[0], 3)

    def test_loop(self):
        pass

    def test_add_exact_coords(self):
        poly_1 = Polygon(self.coords)
        poly_2 = Polygon(self.coords_square,loop=True)
        poly_sum = Polygon.add_exact_coords(poly_1, poly_2)
        np_tests.assert_allclose(poly_sum.t, np.sort(np.concatenate((np.array([0, 1., 6, 10, 14]) / 14.,np.arange(4)/4))))

        poly_1 = Polygon(self.coords_triangle,loop=True)
        poly_2 = Polygon(self.coords_square,loop=True)
        poly_sum = Polygon.add_exact_coords(poly_1,poly_2)
        coord_at_half = poly_sum.eval_coords(.5)
        self.assertAlmostEqual(coord_at_half[0],)
        a = 1

    def test_add_coords_loops(self):
        round_square = Path(self.coords_square,loop=True)
        round_triangle = Path(self.coords_triangle,loop=True)
        poly_sum = Polygon.add_exact_coords(round_square,round_triangle)

    def test_concat(self):
        pass


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