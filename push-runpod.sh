#!/usr/bin/env bash

# push-runpod.sh: push the working tree to the pod, one way.

set -euo pipefail
rsync -az --chown=root:root --delete \
  --exclude='.git/' --exclude='build*/' --exclude='data/' --exclude='.*.swp' \
  ./ runpod:/workspace/toy_llama/
