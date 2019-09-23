"""setup.py script for warp-ctc TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
import warnings
from setuptools.command.build_ext import build_ext as orig_build_ext
from distutils.version import LooseVersion

# We need to import tensorflow to find where its include directory is.
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError("Tensorflow must be installed to build the tensorflow wrapper.")

if "CUDA_HOME" not in os.environ:
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h",
          file=sys.stderr)
    enable_gpu = False
else:
    enable_gpu = True

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

warp_ctc_path = "../build"
if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc"+lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path),
          file=sys.stderr)
    sys.exit(1)

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = tf.sysconfig.get_lib() # os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
warp_ctc_includes = [os.path.join(root_path, '../include')]
include_dirs = tf_includes + warp_ctc_includes

if LooseVersion(tf.__version__) >= LooseVersion('1.4'):
    nsync_dir = '../../external/nsync/public'
    if LooseVersion(tf.__version__) >= LooseVersion('1.10'):
        nsync_dir = 'external/nsync/public'
    include_dirs += [os.path.join(tf_include, nsync_dir)]

if os.getenv("TF_CXX11_ABI") is not None:
    TF_CXX11_ABI = os.getenv("TF_CXX11_ABI")
else:
    warnings.warn("Assuming tensorflow was compiled without C++11 ABI. "
                  "It is generally true if you are using binary pip package. "
                  "If you compiled tensorflow from source with gcc >= 5 and didn't set "
                  "-D_GLIBCXX_USE_CXX11_ABI=0 during compilation, you need to set "
                  "environment variable TF_CXX11_ABI=1 when compiling this bindings. "
                  "Also be sure to touch some files in src to trigger recompilation. "
                  "Also, you need to set (or unsed) this environment variable if getting "
                  "undefined symbol: _ZN10tensorflow... errors")
    TF_CXX11_ABI = "0"

extra_compile_args = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI]
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if LooseVersion(tf.__version__) >= LooseVersion('1.4'):
    if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
        extra_link_args = ['-L' + tf_src_dir, '-ltensorflow_framework']

if (enable_gpu):
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
    include_dirs += [os.path.join(os.environ["CUDA_HOME"], 'include')]

    # mimic tensorflow cuda include setup so that their include command work
    if not os.path.exists(os.path.join(root_path, "include")):
        os.mkdir(os.path.join(root_path, "include"))

    cuda_inc_path = os.path.join(root_path, "include/cuda")
    if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != os.environ["CUDA_HOME"]:
        if os.path.exists(cuda_inc_path):
            os.remove(cuda_inc_path)
        os.symlink(os.environ["CUDA_HOME"], cuda_inc_path)
    include_dirs += [os.path.join(root_path, 'include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
    if not os.path.exists(loc):
        print(("Could not find file or directory {}.\n"
               "Check your environment variables and paths?").format(loc),
              file=sys.stderr)
        sys.exit(1)

lib_srcs = ['src/ctc_op_kernel.cc', 'src/warpctc_op.cc']

ext = setuptools.Extension('warpctc_tensorflow.kernels',
                           sources = lib_srcs,
                           language = 'c++',
                           include_dirs = include_dirs,
                           library_dirs = [warp_ctc_path],
                           runtime_library_dirs = [os.path.realpath(warp_ctc_path)],
                           libraries = ['warpctc', 'tensorflow_framework'],
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args)

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

def discover_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

# Read the README.md file for the long description. This lets us avoid
# duplicating the package description in multiple places in the source.
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
with open(README_PATH, "r") as handle:
    # Extract everything between the first set of ## headlines
    LONG_DESCRIPTION = re.search("#.*([^#]*)##", handle.read()).group(1).strip()

setuptools.setup(
    name = "warpctc_tensorflow",
    version = "0.1",
    description = "TensorFlow wrapper for warp-ctc",
    long_description = LONG_DESCRIPTION,
    url = "https://github.com/baidu-research/warp-ctc",
    author = "Jared Casper",
    author_email = "jared.casper@baidu.com",
    license = "Apache",
    packages = ["warpctc_tensorflow"],
    ext_modules = [ext],
    cmdclass = {'build_ext': build_tf_ext},
    test_suite = 'setup.discover_test_suite',
)
