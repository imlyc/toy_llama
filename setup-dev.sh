#!/bin/bash

# init git submodules
git submodule update --init --recursive --depth 1

# Download 1B model
mkdir -p data/models
pushd data/models
wget https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf
popd

echo "Done"
