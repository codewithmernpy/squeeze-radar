"""Module entrypoint for `python -m squeezeradar`."""

import sys

from squeezeradar.cli import main

if __name__ == "__main__":
    sys.exit(main())
