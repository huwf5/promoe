# Install

## Docker (llama.cpp)

First build docker image and create container

```shell
cd docker
docker build -t moe-cache-llama.cpp -f Dockerfile.llama.cpp .
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true --ipc=host --name moe-cache-llama.cpp-demo -it moe-cache-llama.cpp  bash
```

It it recommended to map host directories with large volume or code repos into container:

```bash
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true -v <huggingface model filder>:/root/.cache/huggingface  -v <your home>:/code  --ipc=host --name moe-cache-llama.cpp-demo -it moe-cache-llama.cpp  bash
```

Then clone and build related repos

```bash
mkdir -p /code
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/transformers.git             && cd transformers            && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/eval-helper.git              && cd eval-helper             && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/expert-selection-tracer.git  && cd expert-selection-tracer && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache-scripts.git
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/llama.cpp.git
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache.git
# build sparse-llm-cache (python ver.)
cd /code/sparse-llm-cache
pip install -e . --no-build-isolation
# build sparse-llm-cache (cpp ver.)
cd /code/sparse-llm-cache
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES="native"
cmake --build build --config Release --parallel 40
# build llama.cpp
cd /code/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES="native"
cmake --build build --config Release --parallel 40
```

## Docker (Transformers)

First build docker image and create container

```shell
cd docker
docker build -t moe-cache-transformers -f Dockerfile.transformers .
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true --ipc=host --name moe-cache-trans-demo -it moe-cache-transformers  bash
```

It it recommended to map host directories with large volume or code repos into container:

```bash
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true -v <huggingface model filder>:/root/.cache/huggingface  -v <your home>:/code  --ipc=host --name moe-cache-trans-demo -it moe-cache-transformers  bash
```

Then clone and build related repos

```bash
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/transformers.git             && cd transformers            && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/eval-helper.git              && cd eval-helper             && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache.git         && cd sparse-llm-cache        && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/expert-selection-tracer.git  && cd expert-selection-tracer && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache-scripts.git
```
