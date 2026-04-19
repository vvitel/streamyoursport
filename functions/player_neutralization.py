#récupérer les détections des joueurs et de la balle
def get_detection(res):
    if res.boxes is None or len(res.boxes) == 0:
        return [], [], [], []

    mask_player = res.boxes.cls.int() == 0
    box_player = res.boxes.xywh[mask_player].tolist()
    confidence_player = res.boxes.conf[mask_player].tolist()

    mask_ball = res.boxes.cls.int() == 1
    box_ball = res.boxes.xywh[mask_ball].tolist()
    confidence_ball = res.boxes.conf[mask_ball].tolist()

    return box_player, confidence_player, box_ball, confidence_ball

#retirer les détections de balles dans la zone d'un joueur
def clear_detection(box_ball, confidence_ball, box_player):
    good_ball_detect = []

    for (x_b, y_b, w_b, h_b), conf_b in zip(box_ball, confidence_ball):
        inside_player = False

        for x_p, y_p, w_p, h_p in box_player:
            
            padding = 15
            x1 = x_p - 0.5 * w_p - padding
            x2 = x_p + 0.5 * w_p + padding
            y1 = y_p - 0.5 * h_p - padding
            y2 = y_p + 0.5 * h_p + padding

            if x1 < x_b < x2 and y1 < y_b < y2:
                inside_player = True
                break

        if not inside_player:
            good_ball_detect.append([x_b, y_b, w_b, h_b, conf_b])

    return good_ball_detect

