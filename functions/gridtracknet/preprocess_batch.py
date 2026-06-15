import cv2
import numpy as np

def preprocess_batch(batch, width, height):
    unit = []

    for frame in batch:
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.moveaxis(frame, -1, 0)  #HWC->CHW

        unit.append(frame[0])  #R
        unit.append(frame[1])  #G
        unit.append(frame[2])  #B

    return unit