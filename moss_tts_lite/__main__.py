"""``python -m moss_tts_lite "text" -o out.wav`` entry point."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
