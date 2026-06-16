import cv2
import time
import queue
import threading
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from torchcodec.decoders import VideoDecoder
from .player_neutralization import neutralize_player
from .engine_export import ensure_engine

BATCH = 8
QUEUE_DEPTH = 4  # nb de lots tampons entre les étages (back-pressure + mémoire bornée)


def _decode_worker(video, step, batch, live, out_q, timing):
    """Étage 1 : décodage (NVDEC pour un fichier, OpenCV pour le live)."""
    try:
        if live:
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
                        out_q.put((idxs, frames))
                        idxs, frames = [], []
                frame_index += 1
            if frames:
                out_q.put((idxs, frames))
            cap.release()
        else:
            decoder = VideoDecoder(video, device="cuda")
            indices = list(range(0, decoder.metadata.num_frames, step))
            for i in range(0, len(indices), batch):
                t = time.perf_counter()
                chunk = indices[i:i + batch]
                frames = decoder.get_frames_at(indices=chunk).data       # [N,3,H,W] RGB GPU
                frames = frames.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]  # -> [N,H,W,3] BGR CPU
                frames = np.ascontiguousarray(frames)
                timing["decode"] += time.perf_counter() - t
                out_q.put((chunk, list(frames)))
    finally:
        out_q.put(None)  # sentinelle de fin


def _human_worker(model_human, in_q, out_q, timing):
    """Étage 2 : détection des joueurs + neutralisation."""
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        idxs, frames = item
        t = time.perf_counter()
        results = model_human(frames, verbose=False)
        frames = [neutralize_player(f, r) for f, r in zip(frames, results)]
        timing["human"] += time.perf_counter() - t
        out_q.put((idxs, frames))


def _ball_worker(model_ball, in_q, data, pbar, timing):
    """Étage 3 : détection de la balle + collecte des détections."""
    while True:
        item = in_q.get()
        if item is None:
            break
        idxs, frames = item
        t = time.perf_counter()
        results = model_ball(frames, verbose=False)
        timing["ball"] += time.perf_counter() - t
        for r, frame_index in zip(results, idxs):
            boxes = r.boxes.xywh.tolist()
            confs = r.boxes.conf.tolist()
            for b, c in zip(boxes, confs):
                b.extend([c, frame_index])
            data.extend(boxes)
        pbar.update(len(frames))


def apply_yolo(path_to_model, video, step, batch=BATCH):
    #live ou mp4
    live = (video == "live")
    if live:
        video = 0

    #charger les modèles en TensorRT FP16 (export auto si le .engine manque)
    model_ball = YOLO(ensure_engine(path_to_model, batch=batch), task="detect")
    model_human = YOLO(ensure_engine("./weights/best_human.engine", batch=batch), task="detect")

    #pipeline en 3 étages : decode(n+2) || humain(n+1) || balle(n)
    #les appels lourds (NVDEC, inférence, copies) libèrent le GIL -> recouvrement réel
    raw_q = queue.Queue(maxsize=QUEUE_DEPTH)    # decode  -> humain
    neut_q = queue.Queue(maxsize=QUEUE_DEPTH)   # humain  -> balle
    data = []
    timing = {"decode": 0.0, "human": 0.0, "ball": 0.0}

    with tqdm() as pbar:
        decode_t = threading.Thread(
            target=_decode_worker, args=(video, step, batch, live, raw_q, timing), daemon=True
        )
        human_t = threading.Thread(
            target=_human_worker, args=(model_human, raw_q, neut_q, timing), daemon=True
        )
        decode_t.start()
        human_t.start()
        #la balle (étage le plus coûteux) tourne dans le thread principal
        _ball_worker(model_ball, neut_q, data, pbar, timing)
        human_t.join()
        decode_t.join()

    #temps cumulé par étage : l'étage le plus long borne le débit du pipeline
    print(
        "temps cumulé par étage ⏱️  "
        f"decode: {timing['decode']:.1f}s | humain: {timing['human']:.1f}s | balle: {timing['ball']:.1f}s"
    )
    return data
