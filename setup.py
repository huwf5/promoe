import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='sparse_llm_cache',
    ext_modules=[
        CUDAExtension('sparse_llm_cache.cpp_worker', 
            [
                'src/cpp_worker/adapter-llama.cpp',
                'src/cpp_worker/adapter.cpp',
                'src/cpp_worker/logging.cc',
                'src/cpp_worker/model_loader.cpp',
                'src/cpp_worker/prefetcher.cpp',
                'src/cpp_worker/predictor.cpp',
                'src/cpp_worker/utils.cpp',
                'src/cpp_worker/profiler.cpp',
                'src/cpp_worker/cache.cpp',
                'src/cpp_worker/worker.cpp',
                'src/cpp_worker/cuda_helper_func.cu',
            ],
            extra_compile_args={
                'cxx': ['-g', '-fopenmp', '-Wno-sign-compare', '-Wno-attributes', '-DSPARSE_LLM_CACHE_ENABLE_NVTX=1'],
                'nvcc': ['-g'],
            },
            libraries = ['cuda']
        ),
    ],
    include_dirs=[os.path.join(_REPO_ROOT, '3rdparty', 'json', 'single_include')],
    cmdclass={
        'build_ext': BuildExtension
    }
)
