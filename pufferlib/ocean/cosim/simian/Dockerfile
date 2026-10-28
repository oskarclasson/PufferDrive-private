# PufferDrive policy server for simian-adapter, CPU only. Build from the repository root:
#   docker build -f pufferlib/ocean/cosim/simian/Dockerfile -t pufferdrive-policy:local .
# The checkpoint is mounted at runtime: /policy/final_model.pt with its config.yaml next to it.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy "gymnasium==0.29.1" pyyaml "setuptools<81"

WORKDIR /pufferdrive
COPY . .
RUN NO_TRAIN=1 python setup.py build_ext --inplace --force

ENV PYTHONPATH=/pufferdrive
EXPOSE 8710
ENTRYPOINT ["python", "-m", "pufferlib.ocean.cosim.simian.server"]
CMD ["--checkpoint", "/policy/final_model.pt"]
