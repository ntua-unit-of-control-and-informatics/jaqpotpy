# type_enforcer.py
from functools import wraps
import inspect
import typing

def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Type check helper function
        def check_type(value, expected_type):
            origin = typing.get_origin(expected_type)
            args = typing.get_args(expected_type)
            
            if origin is None:
                # Base case: simple type
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type}, got {type(value)}")
            elif origin is typing.Union:
                # Handle Union
                if not any(isinstance(value, arg) for arg in args):
                    raise TypeError(f"Expected one of {args}, got {type(value)}")
            elif origin is list:
                # Handle List
                if not isinstance(value, list):
                    raise TypeError(f"Expected list, got {type(value)}")
                for item in value:
                    check_type(item, args[0])
            elif origin is dict:
                # Handle Dict
                if not isinstance(value, dict):
                    raise TypeError(f"Expected dict, got {type(value)}")
                key_type, value_type = args
                for k, v in value.items():
                    check_type(k, key_type)
                    check_type(v, value_type)
            elif origin is tuple:
                # Handle Tuple
                if not isinstance(value, tuple):
                    raise TypeError(f"Expected tuple, got {type(value)}")
                if len(value) != len(args):
                    raise TypeError(f"Expected tuple of length {len(args)}, got {len(value)}")
                for item, expected in zip(value, args):
                    check_type(item, expected)
            else:
                # Handle other generic types
                if not isinstance(value, origin):
                    raise TypeError(f"Expected {origin}, got {type(value)}")
                for item in value:
                    check_type(item, args[0])

        # Iterate over the parameters and their types
        for name, value in bound_args.arguments.items():
            expected_type = func.__annotations__.get(name)
            if expected_type:
                check_type(value, expected_type)

        return func(*args, **kwargs)
    return wrapper


def enforce_class_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):
            setattr(cls, attr_name, enforce_types(attr_value))
    return cls