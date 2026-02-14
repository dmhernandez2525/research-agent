FROM python:3.11-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

RUN groupadd --system agent \
    && useradd --system --create-home --gid agent --shell /usr/sbin/nologin agent

WORKDIR /app
COPY --from=build --chown=agent:agent /app /app

RUN mkdir -p /app/data /app/reports /tmp \
    && chown -R agent:agent /app /tmp

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER agent

VOLUME ["/app/data", "/app/reports"]

HEALTHCHECK --interval=30s --timeout=8s --start-period=15s --retries=3 \
    CMD ["research-agent", "doctor", "--quiet", "--no-api-probes"]

ENTRYPOINT ["research-agent"]
