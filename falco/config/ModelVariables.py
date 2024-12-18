from falco.config import Object


class ModelVariables(Object):
    """Model variables for FALCO model inputs."""

    def __init__(self, **kwargs):
        self.sbpIndex = 0
        "list index of subband"

        self.wpsbpIndex = 0
        "list index of wavelength"

        self.starIndex = 0
        "list index of star"

        self.zernIndex = 1
        "Noll Zernike"

        self.whichSource = 'star'

        self.x_offset = 0
        "lambda_central/D"

        self.y_offset = 0
        "lambda_central/D"

        super().__init__(**kwargs)
