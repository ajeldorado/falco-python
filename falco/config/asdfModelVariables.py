import falco.util
from falco.util import _spec_arg

class ModelVariables:

    def __init__(self, **kwargs):

        self.sbpIndex = _spec_arg("sbpIndex", kwargs, 0)
        self.whichSource = _spec_arg("whichSource", kwargs, '')
        self.wpsbpIndex = _spec_arg("wpsbpIndex", kwargs, 0)
        self.zernIndex = _spec_arg("zernIndex", kwargs, 0)

    pass
