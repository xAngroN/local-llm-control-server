"""Allow running as ``python -m llamactl``."""

import sys

from llamactl.cli import main

if __name__ == "__main__":
    sys.exit(main())
