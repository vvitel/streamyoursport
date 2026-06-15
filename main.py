import argparse
from functions.retrieve_frames import retrieve_frames
from functions.gridtracknet.build_model_input import build_model_input
from functions.gridtracknet.model_definition import GridTrackNet
from functions.gridtracknet.run_model import run_model
from functions.gridtracknet.decode_to_coordinates import decode_to_coordinates

#définition des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True, type=str)
args = ap.parse_args()
video_path = args.video_path

IMGS_PER_INSTANCE = 5
WIDTH, HEIGHT = 768, 432
GRID_COLS, GRID_ROWS = 48, 27

#récupérer les frames avec les joueurs effacés
lst_frame = retrieve_frames(video_path)
frame_height = lst_frame[0].shape[0]
frame_width = lst_frame[0].shape[1]

seq = [
    lst_frame[i:i+IMGS_PER_INSTANCE]
    for i in range(0, len(lst_frame), IMGS_PER_INSTANCE)
    if len(lst_frame[i:i+IMGS_PER_INSTANCE]) == IMGS_PER_INSTANCE
]

#appliquer gridtracknet
units = build_model_input(seq, WIDTH, HEIGHT)
model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
model.load_weights("./weights/model_weights.h5")
y = run_model(model, units)
coords = decode_to_coordinates(y, frame_width, frame_height, IMGS_PER_INSTANCE, WIDTH, HEIGHT, GRID_ROWS, GRID_COLS)

print(coords)
