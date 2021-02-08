
import os
import sys
import platform
import ez_setup
ez_setup.use_setuptools()

from setuptools import find_packages, setup, Extension

copy_args = sys.argv[1:]

if os.name == 'posix':
    copy_args.append('--user')
    
    if  platform.system() == 'Linux':
        extra_compile_args = ['-fPIC']
        ext_modules = [Extension('proper/libcconv', sources = ['proper/cubic_conv_c.c'], extra_compile_args = extra_compile_args, extra_link_args = ['-shared']),
                       Extension('proper/libcconvthread', sources = ['proper/cubic_conv_threaded_c.c'], extra_compile_args = extra_compile_args, extra_link_args = ['-shared']),
                       Extension('proper/libszoom', sources = ['proper/prop_szoom_c.c'], extra_compile_args = extra_compile_args, extra_link_args = ['-shared'])]
    elif platform.system() == 'Darwin':
        from distutils import sysconfig
        vars = sysconfig.get_config_vars()
        vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
        ext_modules = [Extension('proper/libcconv', sources = ['proper/cubic_conv_c.c'], extra_link_args = ['-shared']),
                       Extension('proper/libcconvthread', sources = ['proper/cubic_conv_threaded_c.c'], extra_link_args = ['-shared']),
                       Extension('proper/libszoom', sources = ['proper/prop_szoom_c.c'], extra_link_args = ['-shared'])]
else:
    ext_modules = []

setup(
      name="PyPROPER3",
      version = "3.2.4",
      packages=find_packages(),

      # PROPER uses numpy, scipy and pyfits
      install_requires = ['numpy>=1.8', 'scipy>=0.14', 'astropy>=1.3'],

      # Optional package required
      extras_require = {'pyfftw': ['pyfftw>=0.10']},

      package_data = {
        # If any package contains *.txt, *.rst or *.fits files, include them:
        '': ['*.txt', '*.rst', '*.fits', '*.c', '*.pdf']
      },

      script_args = copy_args,

      author="Navtej Saini, Nikta Amiri, Luis Marchen",
      description="An optical wavefront propagation utility",
      license = "BSD",
      platforms=["any"],
      ext_modules = ext_modules
)
