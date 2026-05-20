import numpy as np

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
        first_frame = block[:, FRAME].min()
        last_frame = block[:, FRAME].max()

        #répartition sur la toile
        cell_size = 20
        points = block[:, [X, Y]].astype(int)
        gx = points[:, 0] // cell_size
        gy = points[:, 1] // cell_size
        occupied_cell = len(set(zip(gx, gy)))
        print(occupied_cell)

        rows.append([i, first_frame, last_frame, occupied_cell])

    return data