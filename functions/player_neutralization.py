def neutralize_player(frame, human_result, padding=15):
    #parcourir les détections d'humains de cette frame (déjà calculées en batch)
    h, w = frame.shape[:2]
    for box in human_result.boxes:
        cls = int(box.cls[0])

        #si on détecte un humain (classe 0)
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            #ajouter une zone de confiance
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

            #effacer les joueurs
            frame[y1:y2, x1:x2] = 0

    return frame
