"""Entry point -- `python -m adia.api` runs the API. The same command for local development and
production: every setting below is read from the environment, defaulting to values that work
unchanged on a developer's machine, so there is no separate dev-only vs. prod-only launcher to
keep in sync. See `.env.example` for the variables this reads.
"""

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("ADIA_API_PORT") or "8000")
    host = os.environ.get("ADIA_API_HOST", "0.0.0.0")
    reload = os.environ.get("ADIA_API_RELOAD", "").strip().lower() == "true"
    uvicorn.run("adia.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
