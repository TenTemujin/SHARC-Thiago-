import os
import glob
import re

directory = r"D:\SHARC\SHARC_GPU\SHARC-Thiago-\sharc\propagation"
files = glob.glob(os.path.join(directory, "propagation_*.py"))

bad_code = """import sys
try:
    import cupy as _cp
    _ArrayType = (np.ndarray, _cp.ndarray)
except ImportError:
    _ArrayType = (np.ndarray,)"""

good_code = """import sys
import numpy as _np
try:
    import cupy as _cp
    _ArrayType = (_np.ndarray, _cp.ndarray)
except ImportError:
    _ArrayType = (_np.ndarray,)"""

modified_count = 0
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if bad_code in content:
        content = content.replace(bad_code, good_code)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        modified_count += 1
        print(f"Fixed: {file_path}")

print(f"Fixed {modified_count} files.")
