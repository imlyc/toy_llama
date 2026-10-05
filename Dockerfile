FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
ENTRYPOINT []
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ git cmake \
    && rm -rf /var/lib/apt/lists/*
