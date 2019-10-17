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


    def test_CanImportModelParameters(cls):
        pass

    def test_CanCreateMpObject(cls):
   
        """ Test creation of MP object """
        mp = falco.config.ModelParameters()
        assert mp is not None

        pass
