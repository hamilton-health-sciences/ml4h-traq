import contextlib
from concurrent.futures import Future


class DummyExecutor(contextlib.nullcontext):
    def submit(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        fut = Future()
        fut.set_result(result)

        return fut
