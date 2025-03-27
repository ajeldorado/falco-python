from abc import ABC, abstractmethod



class TestbedInterface(ABC):
    '''An abstract class whose sole method is implemented by the testbed'''
    
    @abstractmethod
    def get_sbp_image(self,si):
        pass
    