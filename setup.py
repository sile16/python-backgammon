from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
from Cython.Build import cythonize
import numpy as np
import os
import subprocess
import sys


pyx_files = ["bg_common.pyx", "bg_board.pyx", "bg_moves.pyx", "bg_game.pyx"]


# Define Extensions for each module
extensions = [
    Extension(
        name.split('.')[0],  # e.g., 'bg_common'
        sources=[os.path.join("python_backgammon", name)],
        include_dirs=[np.get_include(), "python_backgammon"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for name in pyx_files
]

# Custom build_ext command to generate .pyi files after building extensions
class BuildExtWithStubs(build_ext_orig):
    def run(self):
        # Run the standard build_ext command
        super().run()
        # Generate .pyi stubs
        self.generate_stubs()

    def generate_stubs(self):
        """Generate .pyi stub files for the compiled Cython modules."""
        print("\nGenerating .pyi stub files using stubgen...")
        for ext in extensions:
            module_name = ext.name  # e.g., 'bg_common'
            try:
                # Run stubgen as a subprocess
                subprocess.run(["stubgen", "-m", module_name, "-o", "."], check=True)
                print(f"Stub generated for {module_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate stub for {module_name}: {e}")
            except FileNotFoundError:
                print("stubgen not found. Ensure 'mypy' is installed.")
                sys.exit(1)

setup(
    name="python_backgammon",
    version="1.0.0",
    packages=["python_backgammon"],
    description="Python Backgammon with Cython Optimizations",
    author="Matthew Robertson",
    author_email="sile16@gmail.com",
    #package_dir={"python_backgammon": "python_backgammon"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "binding": True,
        },
        include_path=[np.get_include(), "python_backgammon"],
    ),
    cmdclass={"build_ext": BuildExtWithStubs},
    include_dirs=[np.get_include(), "src"],
    zip_safe=False,
    install_requires=[
        "numpy",
        "cython",
    ],
    setup_requires=[
        "numpy",
        "cython",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
    ],
)
