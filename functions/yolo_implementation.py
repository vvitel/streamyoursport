import cv2
from tqdm import tqdm
from ultralytics import YOLO
from .player_neutralization import neutralize_player

def apply_yolo(path_to_model, video, step):
    #live ou mp4
    if video == "live": video = 0

    #charger modèles
    model_ball = YOLO(path_to_model)
    model_ball.to("cuda")
    model_human = YOLO("./weights/best_human.pt")
    model_human.to("cuda")

    #charger la vidéo
    cap = cv2.VideoCapture(video)

    #parcourir chaque image
    data, frame_index = [], 0
    with tqdm() as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            #appliquer le modèle
            if frame_index % step == 0:
                #effacer les joueurs
                frame = neutralize_player(frame, model_human)
                results = model_ball(frame, verbose=False, device=0)

                #enregistrer les détections
                boxes = results[0].boxes.xywh.tolist()
                confs = results[0].boxes.conf.tolist()
                for b, c in zip(boxes, confs):
                    b.extend([c, frame_index])
                data.extend(boxes)
            
            frame_index += 1
            pbar.update(1)
        
    cap.release()
    return data