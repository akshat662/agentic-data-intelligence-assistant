"""Enables `python -m adia` to run the interactive CLI."""

import sys

from adia.cli import main

if __name__ == "__main__":
    sys.exit(main())
