def is_factory(func):
    return hasattr(func, '_is_factory')


def factory_function(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_factory = True
    return wrapper


def initialize_mapper(func, **kwargs):
    if is_factory(func):
        return func(**kwargs)
    return func
