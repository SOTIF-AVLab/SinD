from __future__ import annotations

import pickle
from typing import Any, BinaryIO


def load_legacy_pandas_pickle(handle: BinaryIO) -> Any:
    try:
        return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name != "pandas.core.indexes.numeric":
            raise

    handle.seek(0)
    import pandas as pd

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "pandas.core.indexes.numeric":
                return getattr(pd, name, pd.Index)
            return super().find_class(module, name)

    return _CompatUnpickler(handle).load()
