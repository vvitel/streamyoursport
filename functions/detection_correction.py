import numpy as np

def correct_ball_detection(lst_data, step):
    detection_array = np.array(lst_data)

    #supprimer détections trop proches sur la même frame
    correct_detection = []
    X, Y, CONF, FRAME = 0, 1, 4, 5
    for f in np.unique(detection_array[:, FRAME]):
        detec_frame = detection_array[detection_array[:, FRAME] == f]
        #tri confiance ordre décroissant
        detec_frame_sort = detec_frame[detec_frame[:, CONF].argsort()[::FRAME]]
        pts_xy = detec_frame_sort[:, [X, Y]]

        #calcul des distances
        diff = pts_xy[:, np.newaxis, :] - pts_xy[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
        half_matrix = np.triu(dist_matrix, k=1)

        #identifier les points à supprimer
        indices = np.column_stack(np.where((half_matrix != 0) & (half_matrix < 10)))
        #supprimer les lignes
        if len(indices) > 0:
            mins = np.unique(np.min(indices, axis=1))
            result = np.delete(detec_frame_sort, mins, axis=0)
        else:
            result = detec_frame_sort

        correct_detection.append(result)

    correct_detection = np.concatenate(correct_detection, axis=0)

    #supprimer les détections statiques
    to_remove = set()
    for f in np.unique(correct_detection[:, FRAME]):
        pts1 = correct_detection[correct_detection[:, FRAME] == f][:, [X, Y]]
        pts2 = correct_detection[correct_detection[:, FRAME] == (f + step)][:, [X, Y]]
        if len(pts1) == 0 or len(pts2) == 0: continue

        #calcul des distances
        diff = pts1[:, None, :] - pts2[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        #définition du seuil
        close = dist < 5
        if np.any(close):
            idx1 = np.where(correct_detection[:, FRAME] == f)[0]
            idx2 = np.where(correct_detection[:, FRAME] == (f + step))[0]
            i_idx, j_idx = np.where(close)
            to_remove.update(idx1[i_idx])
            to_remove.update(idx2[j_idx])

    #suppression finale
    if to_remove:
        mask = np.ones(len(correct_detection), dtype=bool)
        mask[list(to_remove)] = False
        correct_detection = correct_detection[mask]

    return correct_detection


