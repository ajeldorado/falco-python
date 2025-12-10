from abc import ABC, abstractmethod

class TestbedInterface(ABC):
    '''A general purpose interface for all testbeds to implement.'''

    @abstractmethod
    def get_sbp_image(self, subband_index):
        pass

