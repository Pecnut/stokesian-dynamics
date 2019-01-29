# all .pyx files in a folder
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'MyProject',
  ext_modules = cythonize(["*.pyx"]),
  include_dirs=[numpy.get_include()],
)
