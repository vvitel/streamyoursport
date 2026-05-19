import numpy as np

def correct_detection(lst_data, step):
    data = np.array(lst_data)
    X, Y, CONF, FRAME = 0, 1, 4, 5

    #garder détection avec confiance > .7
    data = data[data[:, CONF] > 0.7]

    #supprimer les détections statiques
    to_remove = set()
    for f in np.unique(data[:, FRAME]):
        pts1 = data[data[:, FRAME] == f][:, [X, Y]]
        pts2 = data[data[:, FRAME] == (f + step)][:, [X, Y]]
        if len(pts1) == 0 or len(pts2) == 0: continue

        #calcul des distances
        diff = pts1[:, None, :] - pts2[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        #définition du seuil
        close = dist < 5
        if np.any(close):
            idx1 = np.where(data[:, FRAME] == f)[0]
            idx2 = np.where(data[:, FRAME] == (f + step))[0]
            i_idx, j_idx = np.where(close)
            to_remove.update(idx1[i_idx])
            to_remove.update(idx2[j_idx])

    #suppression finale
    if to_remove:
        mask = np.ones(len(data), dtype=bool)
        mask[list(to_remove)] = False
        data = data[mask]

    return data