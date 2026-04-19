import cv2
from tqdm import tqdm
from ultralytics import YOLO
from .player_neutralization import get_detection, clear_detection

def apply_yolo(video, step):
    #live ou mp4
    if video == "live": video = 0

    #charger modèles
    model_player_ball = YOLO("./best_player_ball_s_1_openvino_model/")

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
                result = model_player_ball(frame, verbose=False)

                #récupérer les détections
                xywh_player, _, xywh_ball, conf_ball = get_detection(result[0])
                good_ball = clear_detection(xywh_ball, conf_ball, xywh_player)
                for gb in good_ball: data.append(gb + [frame_index])

            frame_index += 1
            pbar.update(1)

    cap.release()
    return data