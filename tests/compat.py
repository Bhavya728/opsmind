"""
Pytest compatibility shim so tests run with pytest or standalone python.
"""

from contextlib import contextmanager

class RaisesContext:
    def __init__(self, expected_exc):
        self.expected_exc = expected_exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected exception {self.expected_exc} was not raised")
        if not issubclass(exc_type, self.expected_exc):
            return False  # re-raise unexpected exception
        return True  # suppress expected exception

def raises(expected_exc):
    return RaisesContext(expected_exc)
