FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV MLFLOW_TRACKING_URI="https://dagshub.com/saadaltohamy/MLOps-Labs.mlflow"
ENV MODEL_NAME="titanic-base-model"
ENV MODEL_VERSION="2"
ENV PORT="5000"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

CMD mlflow models serve \
  --model-uri "models:/${MODEL_NAME}/${MODEL_VERSION}" \
  --no-conda \
  --host 0.0.0.0 \
  --port ${PORT}
EXPOSE 5000
