"""Object containing pairwise probe properties."""


class Probe:
    """Define the probe properties."""

    def __init__(self):

        self.Npairs = 3  # Number of pair-wise probe pairs to use.
        self.whichDM = 1  # Which DM to use for probing. 1 or 2.
        self.xOffset = 0  # x-offset of the probe center from the DM grid center [actuators]. Use to avoid obscurations.
        self.yOffset = 0  # y-offset of the probe center from the DM grid center [actuators]. Use to avoid obscurations.
        self.rotation = 0  #  rotation angle applied to the probe command [degrees]
        self.gainFudge = 1  # empirical fudge factor to make average probe amplitude match desired value.

        self.radius = 12  # Half-width of the square probed region in the image plane [lambda/D]. (NOTE: Only used for square probes.)
        self.axis = 'alternate'  # Which axis to have the phase discontinuity along. Values can be 'x', 'y', or 'xy' / 'alt' / 'alternate'. The 'alternate' option causes the bad axis to switch between x and y for each subsequent probe pair.  (NOTE: Only used for square probes.)

        self.width = 12  # Width of rectangular probe in focal plane [lambda/D]. (NOTE: Only used for rectangular probes. radius is used instead for a square probed region)
        self.xiOffset = 6  # Horizontal offset from star of rectangular probe in focal plane [lambda/D].  (NOTE: Only used for rectangular probes. No offset for square probed region.)
        self.height = 24  # Height of rectangular probe in focal plane [lambda/D].  (NOTE: Only used for rectangular probes. radius is used instead for a square probed region)
        self.etaOffset = 0  # Vertical offset from star of rectangular probe in focal plane [lambda/D].  (NOTE: Only used for rectangular probes. No offset for square probed region.)

    def show(self):
        print(self.__dict__)
