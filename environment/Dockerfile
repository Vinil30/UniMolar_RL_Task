# Base image pinned by digest so system libraries cannot drift.
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

WORKDIR /app

# Fully resolved, hash-verified dependency closure (torch + CUDA runtime and
# every transitive dependency).
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

# Deterministic cuBLAS workspace; must be set before the CUDA context is
# created, so it lives in the image rather than in user code.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY task_inputs/ /app/task_inputs/
