def neutralize_player(frame, boxes, padding=15):
    """Efface (met à 0) les zones joueurs d'une frame.

    `boxes` : itérable de boîtes xyxy (tenseurs ou listes de 4 valeurs), déjà
    filtrées sur la classe personne — qu'elles viennent d'une détection ou de
    l'interpolation entre deux détections."""
    h, w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        #ajouter une zone de confiance
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

        #effacer les joueurs
        frame[y1:y2, x1:x2] = 0

    return frame
