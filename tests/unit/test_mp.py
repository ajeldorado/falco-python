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
