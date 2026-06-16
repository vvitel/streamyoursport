import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from torchcodec.decoders import VideoDecoder
from .player_neutralization import neutralize_player
from .engine_export import ensure_engine

BATCH = 8


def _batches_nvdec(video, step, batch):
    """Décodage GPU via NVDEC (torchcodec) -> lots de frames BGR (numpy)."""
    decoder = VideoDecoder(video, device="cuda")
    indices = list(range(0, decoder.metadata.num_frames, step))

    for i in range(0, len(indices), batch):
        chunk = indices[i:i + batch]
        # [N, 3, H, W] uint8 RGB sur le GPU
        frames = decoder.get_frames_at(indices=chunk).data
        # -> [N, H, W, 3] sur CPU, puis RGB -> BGR pour rester compatible cv2/ultralytics
        frames = frames.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
        frames = np.ascontiguousarray(frames)
        yield chunk, list(frames)


def _batches_cv2(video, step, batch):
    """Repli CPU (ex: flux webcam "live") : NVDEC ne décode pas une capture brute."""
    cap = cv2.VideoCapture(video)
    frame_index = 0
    idxs, frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step == 0:
            idxs.append(frame_index)
            frames.append(frame)
            if len(frames) == batch:
                yield idxs, frames
                idxs, frames = [], []
        frame_index += 1
    if frames:
        yield idxs, frames
    cap.release()


def apply_yolo(path_to_model, video, step, batch=BATCH):
    #live ou mp4
    live = (video == "live")
    if live:
        video = 0

    #charger les modèles en TensorRT FP16 (export auto si le .engine manque)
    model_ball = YOLO(ensure_engine(path_to_model, batch=batch), task="detect")
    model_human = YOLO(ensure_engine("./weights/best_human.engine", batch=batch), task="detect")

    #choisir le décodeur : NVDEC pour un fichier, OpenCV pour le live
    batches = _batches_cv2(video, step, batch) if live else _batches_nvdec(video, step, batch)

    #parcourir les images par lots
    data = []
    with tqdm() as pbar:
        for idxs, frames in batches:
            #1) détection des joueurs sur tout le lot
            human_results = model_human(frames, verbose=False)
            #2) effacer les joueurs frame par frame
            frames = [neutralize_player(f, r) for f, r in zip(frames, human_results)]
            #3) détection de la balle sur tout le lot
            ball_results = model_ball(frames, verbose=False)

            #4) enregistrer les détections
            for r, frame_index in zip(ball_results, idxs):
                boxes = r.boxes.xywh.tolist()
                confs = r.boxes.conf.tolist()
                for b, c in zip(boxes, confs):
                    b.extend([c, frame_index])
                data.extend(boxes)

            pbar.update(len(frames))

    return data
