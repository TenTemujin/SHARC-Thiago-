from multipledispatch import dispatch
import numpy as np
try:
    import cupy as cp
    _ArrayType = (np.ndarray, cp.ndarray)
except ImportError:
    _ArrayType = (np.ndarray,)

@dispatch(_ArrayType, _ArrayType)
def get_loss(a, b):
    print("Array method called!")

try:
    get_loss(np.array([1]), np.array([2]))
    print("Numpy success!")
    import cupy
    get_loss(cupy.array([1]), cupy.array([2]))
    print("Cupy success!")
except Exception as e:
    print("Error:", type(e), e)
