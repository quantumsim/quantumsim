class State:

    def __init__(self, *args, **kwargs):
        pass

    def probability(self, *indices, axis='z'):
        raise NotImplementedError()
