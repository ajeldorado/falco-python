from dataclasses import dataclass


@dataclass
class Eval:
    """
    A lazily evaluated expression.

    When evaluating, it exposes numpy, falco, and math to the injected code under `np`, `falco`, and `math`,
    and the model parameters object under `mp`.
    """

    globals: any
    """
    The globals exposed to the injected code, in a dictionary.
    Use this to provide libraries, i.e. `{"np": numpy}`
    """

    locals: any
    """
    The locals exposed to the injected code, in a dictionary.
    This can be used to provide recursive access to the parsed config object.
    Modifying the locals after instantiating this class is OK. (and often necessary)
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
            raise ValueError("Circular parameter evaluation dependency detected.")

        self.in_progress = True
        try:
            result = eval(self.code, self.globals, self.locals)
        finally:
            self.in_progress = False
        return result
