import cv2
from ultralytics import YOLO
from .player_neutralization import neutralize_player

def apply_yolo(path_to_model, video, step, path_to_background):
    #live ou mp4
    stream_live = False
    if video == "live":
        video, stream_live = 0, True

    #charger modèles, background et vidéo
    model_ball = YOLO(path_to_model)
    model_human = YOLO("../weights/best_human.pt")
    background = cv2.imread(path_to_background)
    cap = cv2.VideoCapture(video)

    #parcourir chaque image
    data, frame_index = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        #appliquer le modèle
        if frame_index % step == 0:
            #effacer les joueurs
            frame = neutralize_player(background, frame, model_human)
            results = model_ball(frame, stream=stream_live, verbose=False)

        #enregistrer les détections
        for res in results:
            boxes = res.boxes.xywh.tolist()
            confs = res.boxes.conf.tolist()
            for b, c in zip(boxes, confs):
                b.append(c)
                b.append(frame_index)
                data.append(b)
        frame_index += 1
        
    cap.release()
    return data