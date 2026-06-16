import cv2
import time
import queue
import threading
import torch
from tqdm import tqdm
from ultralytics import YOLO
from torchcodec.decoders import VideoDecoder
from .player_neutralization import neutralize_player
from .engine_export import ensure_engine

BATCH = 8
QUEUE_DEPTH = 4  # nb de lots tampons entre les étages (back-pressure + mémoire bornée)


def _gpu_frames_to_bgr(idxs, gpu_frames):
    """[N tenseurs [3,H,W] RGB GPU] -> (idxs, [N images H,W,3 BGR numpy]).

    Le flip RGB->BGR, le permute et la mise en contigu sont faits SUR LE GPU :
    on transfère alors un bloc déjà contigu (un seul DMA propre) au lieu de subir
    des copies CPU à pas négatif (.cpu() non-contigu + [::-1] + ascontiguousarray),
    qui faisaient chuter le débit de ~690 fps à ~65 fps."""
    batch = torch.stack(gpu_frames)                  # [N,3,H,W] RGB GPU
    batch = batch.flip(1).permute(0, 2, 3, 1)        # [N,H,W,3] BGR GPU (vue)
    arr = batch.contiguous().cpu().numpy()           # contigu sur GPU -> transfert direct
    return idxs, list(arr)


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
            #décodage SÉQUENTIEL (forward-only) = chemin rapide de NVDEC/torchcodec.
            #on parcourt toutes les frames dans l'ordre et on ne garde qu'une frame sur `step`.
            #(éviter get_frames_at sur des indices épars : ça reseek depuis les keyframes)
            decoder = VideoDecoder(video, device="cuda")
            buf_idx, buf_gpu = [], []
            t = time.perf_counter()
            for i, frame in enumerate(decoder):     # frame = [3,H,W] uint8 RGB sur GPU
                if i % step != 0:
                    continue
                buf_idx.append(i)
                buf_gpu.append(frame)
                if len(buf_gpu) == batch:
                    item = _gpu_frames_to_bgr(buf_idx, buf_gpu)
                    timing["decode"] += time.perf_counter() - t
                    out_q.put(item)               # peut bloquer si l'aval est plein (non compté)
                    buf_idx, buf_gpu = [], []
                    t = time.perf_counter()
            if buf_gpu:
                item = _gpu_frames_to_bgr(buf_idx, buf_gpu)
                timing["decode"] += time.perf_counter() - t
                out_q.put(item)
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
