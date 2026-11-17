"""``python -m pyhip.kit <cmd> <module>:<op> ...``: the tuner and table tools
(``pyhip.kit.tune``)."""

import sys

from .tune import main

sys.exit(main())
