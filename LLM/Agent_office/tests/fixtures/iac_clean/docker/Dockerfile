# syntax=docker/dockerfile:1
ARG PY_VERSION=3.12
FROM python:${PY_VERSION}-slim AS builder
ARG PIP_INDEX_TOKEN
USER root
RUN --mount=type=secret,id=pip_token \
    pip wheel --wheel-dir /wheels requests==2.32.3

FROM python:3.12-slim@sha256:1111111111111111111111111111111111111111111111111111111111111111
ENV APP_ENV=production \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /wheels /wheels
ADD app.tar.gz /app/
RUN useradd -u 10001 app
USER 10001:10001
CMD ["python", "-m", "app"]
