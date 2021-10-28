import os
import pickle
from typing import Any


def save_object(obj: Any,
                file_name,
                pickle_format=2):
    """Save a Python object by pickling it.

Unless specifically overridden, we want to save it in Pickle format=2 since this
will allow other Python2 executables to load the resulting Pickle. When we want
to completely remove Python2 backward-compatibility, we can bump it up to 3. We
should never use pickle.HIGHEST_PROTOCOL as far as possible if the resulting
file is manifested or used, external to the system.
    """
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle_format)


def load_object(file_name) -> Any:
    """Load a Python object from pickle.
    """
    file_name = os.path.abspath(file_name)
    with open(file_name, 'rb') as f:
        loaded = pickle.load(f)
    return loaded
