import cv2
from ultralytics import YOLO
from .player_neutralization import neutralize_player

def retrieve_frames(video):
    #charger la vidéo
    cap = cv2.VideoCapture(video)

    #charger modèles
    model_yolo_human = YOLO("./weights/best_human.pt")
    #parcourir chaque image
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = neutralize_player(frame, model_yolo_human)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
    return frame_list