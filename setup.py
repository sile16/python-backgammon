from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

#run with python3 setup.py build_ext

# Set build directories
build_dir = "build"
cython_build_dir = os.path.join(build_dir, "cython")  # For .pyx -> .c
temp_dir = os.path.join(build_dir, "temp")  # For intermediate files
lib_dir = os.path.join(build_dir, "lib")    # For final .so files

# Ensure build directories exist
for d in [build_dir, cython_build_dir, temp_dir, lib_dir]:
    os.makedirs(d, exist_ok=True)

# Define source files and their build locations
pyx_files = ["bg_common.pyx", "bg_board.pyx", "bg_moves.pyx", "bg_game.pyx"]
extensions = [
    Extension(
        name.replace(".pyx", ""),  # Extension name without .pyx
        sources=[os.path.join(cython_build_dir, name.replace(".pyx", ".c"))],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for name in pyx_files
]

setup(
    ext_modules=cythonize(
        pyx_files,
        compiler_directives={
            "language_level": "3",
            "binding": True,
        },
        build_dir=cython_build_dir,  # Directory for generated .c files
    ),
    include_dirs=[np.get_include()],
    options={
        "build": {
            "build_base": build_dir,
            "build_temp": temp_dir,
        },
        "build_ext": {
            "build_lib": lib_dir,
        },
    },
)