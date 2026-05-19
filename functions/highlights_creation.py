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
        length = block.shape[0]
        std_x = block[:, X].std()
        std_y = block[:, Y].std()
        rows.append([i, first_frame, last_frame, length, std_x, std_y])

        import matplotlib.pyplot as plt
        plt.scatter(block[:, X], block[:, Y])
        plt.xlim(0, 1920)
        plt.ylim(1080, 0)
        plt.show()

    #choix des blocs pertinents
    arr_block = np.array(rows)
    total = arr_block[:, 4] + arr_block[:, 5]
    top3 = arr_block[np.argsort(total)[-3:]]
    
    print(top3)
    return data