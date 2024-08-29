# Install

## Docker

First build docker image and create container

```shell
cd docker
docker build -t moe-cache-transformers -f Dockerfile.transformers .
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true --ipc=host --name moe-cache-demo -it moe-cache-transformers  bash
```

It it recommended to map host directories with large volume or code repos into container:

```bash
docker run --runtime nvidia --gpus all  --shm-size=200g  --ulimit memlock=-1 --ulimit core=0  --privileged=true -v <huggingface model filder>:/root/.cache/huggingface  -v <your home>:/code  --ipc=host --name moe-cache-demo -it moe-cache-transformers  bash
```

Then clone and build related repos

```bash
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/transformers.git             && cd transformers            && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/eval-helper.git              && cd eval-helper             && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache.git         && cd sparse-llm-cache        && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/expert-selection-tracer.git  && cd expert-selection-tracer && pip install -e . --no-build-isolation
cd /code && git clone git@ipads.se.sjtu.edu.cn:sparsellm/sparse-llm-cache-scripts.git
```
