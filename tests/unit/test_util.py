import pytest
import numpy as np

import falco


class TestUtils:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_sind(cls):
        ret = falco.util.sind(0)
        assert ret == pytest.approx(0)
        ret = falco.util.sind(90)
        assert ret == pytest.approx(1)
        ret = falco.util.sind(180)
        assert ret == pytest.approx(0)
        ret = falco.util.sind(270)
        assert ret == pytest.approx(-1)
        ret = falco.util.sind(360)
        assert ret == pytest.approx(0)
        ret = falco.util.sind(45)
        assert ret != pytest.approx(0)

    @pytest.mark.parametrize("test_input, expected", [
        (0, 1),
        (90, 0),
        (180, -1),
        (270, 0),
        (360, 1),
    ])
    def test_cosd(cls, test_input, expected):
        ret = falco.util.cosd(test_input)
        assert ret == pytest.approx(expected)

    @pytest.mark.parametrize("test_input, expected", [
        (2, 1.0),
        (100, 7.0),
        (-2, 1.0),
        (-100, 7.0),
        (0, 0),
    ])
    def test_nextpow2(cls, test_input, expected):
        ret = falco.util.nextpow2(test_input)
        assert ret == expected

    @pytest.mark.parametrize("test_input, expected", [
        (0, 0.0),
        (5, 6.0),
        (-2, -2),
        (-1, 0.0),
    ])
    def test_ceil_even(cls, test_input, expected):
        ret = falco.util.ceil_even(test_input)
        assert ret % 2 == 0
        assert ret == expected

    @pytest.mark.parametrize("test_input, expected", [
        (6, 7.0),
        (5, 5.0),
        (-2, -1),
        (-1, -1.0),
    ])
    def test_ceil_odd(cls, test_input, expected):
        ret = falco.util.ceil_odd(test_input)
        assert ret % 2 != 0
        assert ret == expected

    def test_pad_crop(cls):
        test_input = np.zeros((10, 10))
        ret = falco.util.pad_crop(test_input, 20)
        assert ret.shape[0] == 20
        assert ret.shape[1] == 20

        test_input = np.zeros((5, 6))
        ret = falco.util.pad_crop(test_input, (11, 12))
        assert ret.shape[0] == 11
        assert ret.shape[1] == 12

    def test_allcomb(cls):
        in_a = 'abc'
        in_b = 'XY'

        a = falco.util.allcomb(in_a, in_b)

        assert len(a) == len(in_a) * len(in_b)
        assert type(a[0]) is tuple

        count = 0
        for i, d in enumerate(in_a):
            for j, c in enumerate(in_b):
                assert c in a[count] and d in a[count]
                count += 1

    def test__spec_arg(cls):
        
        kwargs = {"logGmin": 545}
        val = falco.util._spec_arg("logGmin", kwargs, -6)

        assert val == 545

    def test__spec_argi_unfoundArg(cls):
        
        kwargs = {"blahblah": 545}
        val = falco.util._spec_arg("logGmin", kwargs, -6)

        assert val == -6

    def test_broadcast(cls):
        a = np.zeros((10,))
        x, y = falco.util.broadcast(a)
        assert x.shape[0] == 1
        assert x.shape[1] == 10

        assert y.shape[0] == 10
        assert y.shape[1] == 1

    def test_radial_grid(cls):
        pass

    def test_radial_grid_squared(cls):
        pass

    def test_create_axis(cls):
        pass

    def test_falco_compute_thput(cls):
        pass
