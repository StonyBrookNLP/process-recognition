__author__ = 'chetannaik'


class Memoize:
    def __init__(self, func):
        self.func = func
        self._cache = {}

    def __call__(self, *args, **kwargs):
        # memoization_key constructed from the function name and arguments
        memoization_key = self._convert_args_to_hash(args, kwargs)
        if memoization_key not in self._cache:
            self._cache[memoization_key] = self.func(*args, **kwargs)
        return self._cache[memoization_key]

    def __repr__(self):
        return self.func.__doc__

    def _convert_args_to_hash(self, args, kwargs):
        return hash(
            self.func.__name__ + str(args) + str(sorted(kwargs.items())))
