from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Set the build directory
build_dir = "build"

# Ensure build directory exists
os.makedirs(build_dir, exist_ok=True)

extensions = [
    Extension(
        "bg_common",
        ["bg_common.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "bg_board",
        ["bg_board.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "bg_moves",
        ["bg_moves.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "bg_game",
        ["bg_game.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "binding": True,
        }
    ),
    include_dirs=[np.get_include()],
    options={
        "build": {"build_base": build_dir}  # Redirect build output
    },
)