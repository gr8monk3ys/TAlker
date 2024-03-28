import collections, functools, logging

PostInfo = collections.namedtuple("PostInfo", ["username", "text", "post_id", "status"])

class ignore_error:
    def __init__(self, *error_types):
        self.error_types = error_types

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.error_types as e:
                logging.info(f"Ignored error of type {type(e)}: {e}")
        return wrapped
