import sys

import numpy as np


if __name__ == "__main__":
    filename = sys.argv[1]
    filepath = f"lpca_out/{filename}.npz"

    # Load (lazy; arrays are read on access)
    data = np.load(filepath, allow_pickle=False)

    cleaned = {}
    for name in data.files:
        arr = data[name]
        # Works for float arrays; for non-floats, it just leaves them unchanged
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0)  # also handles +/-inf unless you override
        else:
            # if it's object dtype, nan_to_num won't work; try safe per-element cleanup if needed
            if arr.dtype == object:
                arr = np.array([0 if (isinstance(x, float) and np.isnan(x)) else x for x in arr], dtype=object)
        cleaned[name] = arr

    out_path = f"lpca_out/{filename}_cleaned.npz"
    np.savez_compressed(out_path, **cleaned)

