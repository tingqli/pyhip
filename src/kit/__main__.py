"""``python -m pyhip.kit <cmd> <module>:<op> ...``: the tuner and table tools
(``pyhip.kit.tune``: cases, candidates, tune, merge, show, verify, tune-all, compare) and
the A/B and correctness tools (``pyhip.kit.bench``: ab, check)."""

import sys

if sys.argv[1:2] in (["ab"], ["check"]):
    from .bench import main
else:
    from .tune import main

sys.exit(main())
