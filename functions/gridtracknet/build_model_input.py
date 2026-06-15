import numpy as np
from functions.gridtracknet.preprocess_batch import preprocess_batch

def build_model_input(batches, width, height):
    units = []

    for batch in batches:
        unit = preprocess_batch(batch, width, height)
        units.append(unit)

    units = np.asarray(units).astype(np.float32)
    units /= 255.0

    return units