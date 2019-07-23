import os
from quantumsim import Setup

test_dir = os.path.dirname(os.path.abspath(__file__))


class TestSetup:
    def test_setup_load(self):
        _ = Setup.from_file(os.path.join(test_dir,
                            'time_aware_setup_with_defaults.yaml'))
