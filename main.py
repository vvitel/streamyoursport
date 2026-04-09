import argparse
from functions.yolo_implementation import apply_yolo
from functions.detection_correction import correct_detection

#définition des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_path", "--model_path", required=True, type=str)
ap.add_argument("-video_path", "--video_path", required=True, type=str)
ap.add_argument("-frame_step", "--frame_step", required=True, type=int)
args = ap.parse_args()

model_path, video_path = args.model_path, args.video_path
frame_step = args.frame_step

#effacer les joueurs et appliquer le modèle
lst_detection = apply_yolo(model_path, video_path, frame_step)
print("the model made its predictions \U0001F52E")

#corriger les détections
df_detection = correct_detection(lst_detection, frame_step)
print("the detections have been filtered \U0001F9FD")
