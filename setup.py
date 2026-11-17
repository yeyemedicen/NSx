'''NSx: fractional-step incompressible Navier-Stokes (ALE, Windkessel outlets) on DOLFINx.

    pip install git+https://github.com/yeyemedicen/NSx.git      # or, from a clone:  pip install -e .
DOLFINx, mpi4py and petsc4py come from the conda environment, never from pip.
'''
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
about = {}
exec(open(os.path.join(here, 'nsx', '__init__.py')).read(), about)
readme = os.path.join(here, 'README.rst')

setup(
    name='nsx',
    version=about['__version__'],
    packages=find_packages(include=['nsx', 'nsx.*']),
    description='Navier-Stokes FEM solver',
    long_description=open(readme, encoding='utf-8').read() if os.path.exists(readme) else '',
    author='Jeremias Garay',
    author_email='jeremias.garay@usm.cl',
    url='https://github.com/yeyemedicen/NSx',
    license='MIT',
    python_requires='>=3.9',
    install_requires=[
        'common @ git+https://github.com/yeyemedicen/Common.git',
        'numpy', 'scipy', 'h5py', 'tqdm', 'packaging',
    ],
    extras_require={'vis': ['pyvista']},

)
