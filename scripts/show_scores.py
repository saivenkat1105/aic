#!/usr/bin/env python3

import sys

from tracking_tools import main


if __name__ == "__main__":
    sys.exit(main(["show-scores", *sys.argv[1:]]))
