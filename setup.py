import os
from setuptools import setup

from nsx import __version__

here = os.path.abspath(os.path.dirname(__file__))

readme_path = os.path.join(here, 'README.rst')
long_description = open(readme_path, encoding='utf-8').read() if os.path.exists(readme_path) else ''

setup(
    name='nsx',
    packages=['nsx'],
    version=__version__,
    description='Navier-Stokes FEM solver',
    long_description=long_description,
    author='Jeremias Garay',
    author_email='jeremias.garay@usm.cl',
    url='asd',
    # requires=['numpy (>=1.7)', 'scipy (>=0.13)'],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
