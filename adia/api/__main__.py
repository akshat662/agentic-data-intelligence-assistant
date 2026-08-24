"""Dev entry point -- `python -m adia.api` runs the API with uvicorn's dev server."""

import uvicorn


def main() -> None:
    uvicorn.run("adia.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
