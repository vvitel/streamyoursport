import torch


def _centers(b):
    return torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], dim=1)


def interpolate_boxes(boxes_a, boxes_b, max_dist=120.0):
    """Interpole les boîtes joueurs d'une frame intermédiaire entre deux détections.

    `boxes_a`, `boxes_b` : tenseurs [n, 4] (xyxy) des deux frames détectées
    encadrant la frame à reconstruire. On apparie chaque joueur de A au plus
    proche de B (distance des centres < `max_dist`) et on prend la position
    médiane. Les joueurs non appariés (apparus / disparus) sont conservés tels
    quels : on préfère sur-masquer plutôt que laisser un joueur visible.

    Retourne une liste de boîtes [4] (xyxy) — suffisant pour la neutralisation.
    """
    if len(boxes_a) == 0:
        return list(boxes_b)
    if len(boxes_b) == 0:
        return list(boxes_a)

    ca, cb = _centers(boxes_a), _centers(boxes_b)
    out, used = [], set()
    for i in range(len(boxes_a)):
        dist = torch.sqrt(((cb - ca[i]) ** 2).sum(dim=1))
        j = int(torch.argmin(dist))
        if float(dist[j]) <= max_dist and j not in used:
            out.append((boxes_a[i] + boxes_b[j]) / 2.0)  # position médiane
            used.add(j)
        else:
            out.append(boxes_a[i])  # joueur disparu -> on garde sa boîte

    for j in range(len(boxes_b)):
        if j not in used:
            out.append(boxes_b[j])  # joueur apparu -> on garde sa boîte

    return out
