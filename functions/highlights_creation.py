import numpy as np
from scipy.spatial import ConvexHull

def create_highlights(data, nb_frame_to_separate):
    X, Y, FRAME, BLOCK  = 0, 1, 5, -1
    data = data[np.argsort(data[:, FRAME])]
    frames = data[:, FRAME]

    #déterminer les blocs
    diff = np.diff(frames, prepend=frames[0])
    new_block = diff >= nb_frame_to_separate
    blocks = np.cumsum(new_block)
    data = np.column_stack((data, blocks))

    #récupérer les infos par bloc
    rows = []
    for i in np.unique(data[:, BLOCK]):
        block = data[data[:, BLOCK] == i]
        block = data[data[:, BLOCK] == i]
        first_frame = block[:, FRAME].min()
        last_frame = block[:, FRAME].max()

        points = block[:, [X, Y]]
        nb_points = len(points)
        #enveloppe convexe
        if nb_points >= 3:
            perimeter = ConvexHull(points).area
            rows.append([i, first_frame, last_frame, nb_points, perimeter])
        
    array = np.array(rows)
    sorted_array = array[np.argsort(-array[:, -1])]
    return sorted_array[:10]