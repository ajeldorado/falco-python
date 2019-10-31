import numpy as np
import pytest
import sys
sys.path.insert(0,"../")
import falco

class TestModelParametersClass:

    @classmethod
    def setup_class(cls):
        print("Setup TestClass!")


    @classmethod
    def teardown_class(cls):
        print("Teardown TestClass!")


    def setup_method(self, method):
        if method == self.test_CanCreateMpObject:
            print("\nSetting up for test_CanCreateMpObject...")
        else:
            print("\nSetting up Unknown test!")


    def teardown_method(self, method):
        if method == self.test_CanCreateMpObject:
            print("\nTearing down for test_CanCreateMpObject...")
        else:
            print("\nTearing down Unknown test!")


    def test_CanCreateMpObject(cls):
   
        """ Test creation of MP object """
        mp = falco.config.ModelParameters()
        assert mp is not None

class TestUtils:
    @classmethod
    def setup_class(cls):
        pass


    @classmethod
    def teardown_class(cls):
        pass


    def test_sind(cls):
        ret = falco.utils.sind(0)
        assert ret == pytest.approx(0)
        ret = falco.utils.sind(90)
        assert ret == pytest.approx(1)
        ret = falco.utils.sind(180)
        assert ret == pytest.approx(0)
        ret = falco.utils.sind(270)
        assert ret == pytest.approx(-1) 
        ret = falco.utils.sind(360)
        assert ret == pytest.approx(0)
        ret = falco.utils.sind(45)
        assert ret != pytest.approx(0)


    @pytest.mark.parametrize("test_input, expected", [(0, 1), (90, 0), (180, -1), (270, 0), (360, 1)])
    def test_cosd(cls, test_input, expected):
        ret = falco.utils.cosd(test_input)
        assert ret == pytest.approx(expected)

    @pytest.mark.parametrize("test_input, expected", [(2, 1.0), (100, 7.0), (-2, 1.0), (-100, 7.0), (0, -np.inf)])
    def test_nextpow2(cls, test_input, expected):
        ret = falco.utils.nextpow2(test_input)
        assert ret == expected

    @pytest.mark.parametrize("test_input, expected", [(0, 0.0), (5, 6.0), (-2, -2), (-1, 0.0)])
    def test_ceil_even(cls, test_input, expected):
        ret = falco.utils.ceil_even(test_input)
        assert ret % 2 == 0
        assert ret == expected;

    @pytest.mark.parametrize("test_input, expected", [(6, 7.0), (5, 5.0), (-2, -1), (-1, -1.0)])
    def test_ceil_odd(cls, test_input, expected):
        ret = falco.utils.ceil_odd(test_input)
        assert ret % 2 != 0
        assert ret == expected;

    def test_padOrCropEven(cls):
        test_input = np.zeros((10,10))
        ret = falco.utils.padOrCropEven(test_input, 20)

        assert ret.shape[0] == 20

        test_input = np.zeros((5, 6))
        with pytest.raises(ValueError):
            ret = falco.utils.padOrCropEven(test_input, 20)
        test_input = np.zeros((8, 6))
        with pytest.raises(ValueError):
            ret = falco.utils.padOrCropEven(test_input, 20)


    def test_allcomb(cls):
        in_a = 'abc'
        in_b = 'XY'

        a = falco.utils.allcomb(in_a, in_b)

        assert len(a) == len(in_a) * len(in_b)
        assert type(a[0]) is tuple

        count = 0
        for i, d in enumerate(in_a):
            for j, c in enumerate(in_b):
                assert c in a[count] and d in a[count]
                count += 1 
        
        pass

    def test__spec_arg(cls):
        pass

    def test_broadcast(cls):
        pass

    def test_radial_grid(cls):
        pass

    def test_radial_grid_squared(cls):
        pass

    def test_create_axis(cls):
        pass

    def test_falco_compute_thput(cls):
        pass

def test_test1Example():
    pass

