from dataclasses import dataclass


@dataclass
class Eval:
    """
    A lazily evaluated expression.

    When evaluating, it exposes numpy and falco to the injected code under `np` and `falco`,
    and the model parameters object under `mp`.
    """
    mp: any
    """
    The model parameters object exposed to the injected code.
    
    Modifying the `mp` object after instantiating this class is OK. (and necessary in most cases)
    """

    code: str
    "The injected code to be run lazily."

    in_progress = False
    """
    Whether evaluation is currently in progress.
    
    If a circular dependency happens, this variable is used to detect it.
    """

    def evaluate(self):
        """Runs the code."""
        if self.in_progress:
            raise Exception("Circular parameter evaluation dependency detected.")

        self.in_progress = True

        import falco
        import numpy

        return eval(self.code, {'np': numpy, 'falco': falco}, {'mp': self.mp})
