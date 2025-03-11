# tuner

## Setup

- Build VLLM container with Dockerfile.cpu
- Build dq container (update Dockerfile with the built vllm image)
- Make sure configs are aligned with the test (update containers.json to add the containers built)

## Steps to run

- export HUGGING_FACE_HUB_TOKEN=\<your token\>
- python tuner.py
