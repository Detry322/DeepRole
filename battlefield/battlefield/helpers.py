from functools import wraps

def simple_cache(f):
    cache = {}
    @wraps(f)
    def wrapper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return wrapper
