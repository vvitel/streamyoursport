def neutralize_player(frame, model_human):
    #appliquer le modèle
    results = model_human(frame, verbose=False)

    #parcourir les détections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            #si on détecte un humain (classe 0)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                #ajouter une zone de confiance
                padding = 15
                h, w, _ = frame.shape
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

                #effacer les joueurs
                frame[y1:y2, x1:x2] = 0
    
    return frame