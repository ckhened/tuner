import os
import json
import sys
import argparse

from huggingface_hub import snapshot_download


MODELS_JSON_FILE = os.environ.get('MODELS_JSON_FILE', '/workspace/configs/models.json')
#MODELS_DOWNLOAD_DIR = os.environ.get('MODELS_DOWNLOAD_DIR', '/workspace/models')


def download_model(model):
    try:
        #snapshot_download(repo_id=model, local_dir=MODELS_DOWNLOAD_DIR)
        snapshot_download(repo_id=model)
    except Exception as e:
        print(f"Error downloading {model} with exception {e}")


def quantize_model(model):
    pass


def main():
    models = None

    try:
        with open(MODELS_JSON_FILE, 'r') as f:
            models = json.load(f)
    except Exception as e:
        print(f"Error opening/reading file {MODELS_JSON_FILE}. Exception is {e}")
        sys.exit(-1)

    for model in models:
        download_model(model['model'])
        quantize_model(model['model'])


if __name__ == '__main__':
    main() 
