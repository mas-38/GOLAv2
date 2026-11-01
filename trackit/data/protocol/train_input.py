from collections import namedtuple

_DEFAULT = object()


class TrainData(namedtuple('TrainDataBase', ('input', 'target', 'miscellanies'))):
    """Tuple protocol with per-batch containers, safe for queued prefetching."""
    __slots__ = ()

    def __new__(cls, input=_DEFAULT, target=_DEFAULT, miscellanies=_DEFAULT):
        return super().__new__(cls, {} if input is _DEFAULT else input,
                               {} if target is _DEFAULT else target,
                               {} if miscellanies is _DEFAULT else miscellanies)
