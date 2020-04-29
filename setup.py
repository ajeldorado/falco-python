#!/usr/bin/env python
import os
import subprocess
import sys
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.test import test as TestCommand

try:
    from distutil.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

# Read configuration variables in from setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata['author']
AUTHOR_EMAIL = metadata['author_email']
URL = metadata['url']
LICENSE = metadata['license']
VERSION = metadata['version']


# This next part allows you to build sphinx docs from the package
# main directory with 'python setup.py build_docs'

try:
    from sphinx.cmd.build import build_main
    from sphinx.setup_command import BuildDoc

    class BuildSphinx(BuildDoc):
        """Build Sphinx documentation after compiling C source files"""

        description = 'Build Sphinx documentation'

        def initialize_options(self):
            BuildDoc.initialize_options(self)

        def finalize_options(self):
            BuildDoc.finalize_options(self)

        def run(self):
            build_cmd = self.reinitialize_command('build_ext')
            build_cmd.inplace = 1
            self.run_command('build_ext')
            build_main(['-b', 'html', './docs', './docs/_build/html'])

except ImportError:
    class BuildSphinx(Command):
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            print('!\n! Sphinx is not installed!\n!', file=sys.stderr)
            exit(1)

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['falco/tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=['sphinx_rtd_theme'
    ],
    tests_require=['pytest'],
    packages=find_packages(),
    package_data={},
    cmdclass={
        'test': PyTest,
        'build_docs': BuildSphinx
    },)
