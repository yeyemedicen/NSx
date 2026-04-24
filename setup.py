import os
from distutils.core import setup

from nsx import __version__

'''
    Install with
        $ pip install --user
    or
        $ pip install -e . --user
    for an editable development build (no need to reinstall package after
    modifications).
'''

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='navierstokes',
    packages=['navierstokes'],
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
