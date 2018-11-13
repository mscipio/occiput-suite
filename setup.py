# OCCIPUT SUITE - Computational engine for volumetric imaging
#
# Stefano Pedemonte
# Aalto University
# Harvard University, Martinos Center for Biomedical Imaging
# 2012 - 2014, Helsinki, FI
# 2013 - 2017, Boston, MA, USA
# --> initial development of the original source packages:
#   -- NiftyPy
#	-- iLang
#	-- PetLink
#	-- Interfile
#	-- SimpleWrap
#	-- DisplayNode
#	-- occiput
#
# Michele Scipioni
# University of Pisa
# 2015 - 2018, Pisa, IT
# --> porting to python 3.6
# --> creation of the now-so-called occiput-suite module, which contains all occiput's dependancies
#	  armonized togheter
# --> continuing development of 'occiput' submodule

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import os
import sysconfig
import sys
from glob import glob

from Cython.Build import cythonize, build_ext
from Cython.Distutils import build_ext

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)

petlink32_c_module = Extension('occiput_suite.petlink.petlink32_c',
                               [os.path.join('occiput_suite', 'petlink', 'petlink32_c.c')])
test_simplewrap_module = Extension('occiput_suite.simplewrap.tests.test_simplewrap_c',
                                   [os.path.join('occiput_suite', 'simplewrap', 'tests', 'test_simplewrap_c.c')])
test_matrices_module = Extension('occiput_suite.simplewrap.tests.test_matrices_c',
                                 [os.path.join('occiput_suite', 'simplewrap', 'tests', 'test_matrices_c.c')])

if sys.version_info >= (3,0):
    build_ext_fun = BuildExtWithoutPlatformSuffix
else:
    build_ext_fun = build_ext

setup(
    name='occiput_suite',
    version='0.5.0',
    packages=['occiput_suite', 'occiput_suite.test', 'occiput_suite.ilang', 'occiput_suite.ilang.test',
              'occiput_suite.ilang.webgui', 'occiput_suite.ilang.examples', 'occiput_suite.NiftyPy',
              'occiput_suite.NiftyPy.test', 'occiput_suite.NiftyPy.NiftyRec', 'occiput_suite.NiftyPy.NiftyReg',
              'occiput_suite.NiftyPy.NiftySeg', 'occiput_suite.occiput', 'occiput_suite.occiput.Core',
              'occiput_suite.occiput.DataSources', 'occiput_suite.occiput.DataSources.Synthetic',
              'occiput_suite.occiput.DataSources.FileSources', 'occiput_suite.occiput.Registration',
              'occiput_suite.occiput.Registration.Affine', 'occiput_suite.occiput.Registration.TranslationRotation',
              'occiput_suite.occiput.Visualization', 'occiput_suite.occiput.Classification',
              'occiput_suite.occiput.Reconstruction', 'occiput_suite.occiput.Reconstruction.CT',
              'occiput_suite.occiput.Reconstruction.MR', 'occiput_suite.occiput.Reconstruction.PET',
              'occiput_suite.occiput.Reconstruction.SPECT', 'occiput_suite.occiput.Transformation',
              'occiput_suite.petlink', 'occiput_suite.petlink.tests', 'occiput_suite.petlink.examples',
              'occiput_suite.interfile', 'occiput_suite.interfile.tests', 'occiput_suite.interfile.examples',
              'occiput_suite.notebooks', 'occiput_suite.simplewrap', 'occiput_suite.simplewrap.tests',
              'occiput_suite.simplewrap.examples', 'occiput_suite.DisplayNode', 'occiput_suite.DisplayNode.tests',
              'occiput_suite.DisplayNode.examples'],
    data_files=[
        (os.path.join('occiput_suite', 'DisplayNode', 'static'), glob(os.path.join('DisplayNode', 'static', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'openseadragon'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'openseadragon', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'openseadragon', 'images'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'openseadragon', 'images', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'js'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'js', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'style'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'style', '*.*'))),
        (os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'images'),
         glob(os.path.join('occiput_suite', 'DisplayNode', 'static', 'tipix', 'images', '*.*')))],
    package_data={'occiput_suite.occiput': ['Data/*.pdf','Data/*.png','Data/*.jpg','Data/*.svg','Data/*.nii','Data/*.dcm','Data/*.h5','Data/*.txt','Data/*.dat']},
    cmdclass={'build_ext': build_ext_fun},
    ext_modules=[petlink32_c_module, test_simplewrap_module, test_matrices_module],
    url='https://github.com/mscipio/occiput-suite',
    license='LICENSE.txt',
    author='Michele Scipioni',
    author_email='scipioni.michele@gmail.com',
    description='Tomographic Vision - PET, SPECT, CT, MRI reconstruction and processing.',
    long_description=open('README.md').read(),
    keywords=["PET", "SPECT", "emission tomography", "transmission tomography",
              "tomographic reconstruction", "nuclear magnetic resonance"],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Bio-Informatics"],
    install_requires=[
        "numpy >= 1.12.0",
        "matplotlib >= 1.4.0",
        "ipy_table >= 1.11.0",
        "nibabel >= 2.0.0",
        "pydicom >= 0.9.0",
        "nipy >= 0.3.0",
        "jupyter >= 1.0.0",
        "h5py >= 2.8.0rc1",
        "scipy >= 0.14.0",
        "pillow >= 2.8.0",
        "svgwrite >= 1.1.0",
        "ipython >= 1.0.0",
        "nipy >= 0.4.0"
    ],
    zip_safe=False
)
