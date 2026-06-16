import argparse
import numpy as np
from functions.yolo_implementation import apply_yolo
from functions.detection_correction import correct_detection
from functions.highlights_creation import create_highlights
from functions.detection_visualization import visualize_detection

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
array_detection = correct_detection(lst_detection, frame_step)
print("the detections have been filtered \U0001F9FD")
np.save("./detection.py", array_detection)

#trouver les moments forts
array_block_metric = create_highlights(array_detection, 35)

#visualiser les détections
#visualize_detection(video_path, array_detection)