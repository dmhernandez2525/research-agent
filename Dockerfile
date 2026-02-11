FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
RUN useradd -m -s /bin/false agent
WORKDIR /app
COPY --from=build --chown=agent:agent /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER agent
ENTRYPOINT ["research-agent"]
