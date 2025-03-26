import numpy as np

from falco.config import Object

class Drift(Object):
    """Define the probe properties."""

    def __init__(self, **kwargs):
        self.type = 'rand_walk'
        "Type of drift injected."

        self.whichDM = np.array([1])
        "Which DM to use for drift injection. 1, 2, or [1,2]."

        self.magnitude = 9e-6
        "magnitude of drift injected."

        self.presumed_dm_std = 9e-6
        "drift assumed to be injected, allows for synthetic model-mismatch."


        super().__init__(**kwargs)
