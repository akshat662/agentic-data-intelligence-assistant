# Backend image for adia/api (FastAPI). Not required to deploy to Render/Railway/etc, which
# build directly from pyproject.toml/uv.lock -- this is the portable/self-host fallback (see
# README.md "Deployment"). Secrets are never baked in: only OPENAI_API_KEY and friends are
# injected at container *run* time (`docker run -e ...` / `docker compose`'s env_file), never
# copied into the image -- see .dockerignore.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependencies first, in their own layer, so an app-code-only change doesn't reinstall them.
# README.md is required here too: pyproject.toml's `readme = "README.md"` makes hatchling
# require it to build the project's own metadata, even for --no-install-project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + the dataset(s) this repo ships and registers by default (data/registry.json,
# data/superstore.csv, data/catalog/ -- see .gitignore for exactly what's tracked).
COPY adia/ ./adia/
COPY data/ ./data/
RUN uv sync --frozen --no-dev

ENV ADIA_API_HOST=0.0.0.0
EXPOSE 8000

# --no-sync: the venv was already built above; without this, `uv run` re-syncs on every
# container start, which pulls in the dev dependency group (ruff, pytest, ...) and adds a slow,
# needless network round-trip to every cold start.
CMD ["uv", "run", "--no-sync", "python", "-m", "adia.api"]
