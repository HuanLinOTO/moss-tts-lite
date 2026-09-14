"""``python -m moss_tts_lite "text" -o out."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
