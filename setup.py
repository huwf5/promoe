from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_llm_cache',
    ext_modules=[
        CUDAExtension('sparse_llm_cache.cpp_worker', [
            'src/cpp_worker/adapter.cpp',
            'src/cpp_worker/logging.cc',
            'src/cpp_worker/model_loader.cpp',
            'src/cpp_worker/prefetcher.cpp',
            'src/cpp_worker/predictor.cpp',
            'src/cpp_worker/utils.cpp',
            'src/cpp_worker/profiler.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-g']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
