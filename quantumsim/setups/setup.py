import yaml


class Setup:
    """

    Parameters
    ----------
    setup : dict, str or stream
        Parameters for initialization of setup.
    """
    def __init__(self, setup):
        if isinstance(setup, dict):
            self._setup_dict = setup
        else:
            self._setup_dict = yaml.safe_load(setup)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            return cls(f)

