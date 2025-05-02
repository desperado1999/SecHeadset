from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        name="denoise_for_prototype",
        sources=["denoise_for_prototype.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="denoise_for_prototype",
    ext_modules=cythonize(extensions, annotate=True)
)