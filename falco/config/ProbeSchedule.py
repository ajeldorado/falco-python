"""Object containing scheduled probe properties for each WFSC iteration."""


class ProbeSchedule:
    """Define the probe schedule."""

    def __init__(self):

        self.xOffsetVec = None  # Vector of x-offsets (one value per WFSC iteration) of the probe center from the DM grid center [actuators]
        self.yOffsetVec = None  # Vector of x-offsets (one value per WFSC iteration) of the probe center from the DM grid center [actuators]
        self.rotationVec = None  # Vector of the rotation angle to add to the probes at each WFSC iteration [degrees]
        self.InormProbeVec = None  # Vector of the desired normalized intensity of the probes at each WFSC iteration 

    def show(self):
        print(self.__dict__)
