import cv2
import time
import queue
import threading
import torch
from tqdm import tqdm
from ultralytics import YOLO
from torchcodec.decoders import VideoDecoder
from .player_neutralization import neutralize_player
from .box_interpolation import interpolate_boxes
from .resource_monitor import ResourceMonitor
from .engine_export import ensure_engine

BATCH = 8
QUEUE_DEPTH = 4         # nb de lots tampons entre les étages (back-pressure + mémoire bornée)
HUMAN_DETECT_EVERY = 2  # détecter les joueurs 1 frame sur 2, interpoler l'autre


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


class _HumanStage:
    """Détection des joueurs 1 frame sur `detect_every`, interpolation pour les
    frames intermédiaires, puis neutralisation.

    L'ordre d'émission n'a pas d'importance : chaque détection de balle porte son
    `frame_index`, et le tri se fait en aval (correct_detection / create_highlights).
    On exploite ça pour bufferiser librement. Une frame impaire (interpolée) n'est
    émise qu'une fois ses deux voisines paires détectées.
    """

    def __init__(self, model, batch, detect_every, timing):
        self.model = model
        self.batch = batch
        self.every = detect_every
        self.timing = timing
        self.position = 0          # compteur de frames traitées (après `step`)
        self.boxes_at = {}         # position paire -> boîtes joueurs xyxy (cpu)
        self.held = {}             # position impaire -> (idx, frame) en attente
        self.det_pending = []      # frames paires à détecter (regroupées en batch)
        self.emit_buf = []         # (idx, frame) neutralisées, prêtes à émettre

    def _detect(self):
        """Inférence batchée sur les frames paires en attente."""
        if not self.det_pending:
            return
        items = self.det_pending
        frames = [f for _, _, f in items]
        t = time.perf_counter()
        #classes=[0] : on ne garde que la personne -> NMS allégé (modèle COCO 80 classes)
        results = self.model(frames, classes=[0], verbose=False)
        for (pos, idx, frame), r in zip(items, results):
            boxes = r.boxes.xyxy.cpu()
            self.boxes_at[pos] = boxes
            neutralize_player(frame, boxes)
            self.emit_buf.append((idx, frame))
        self.timing["human"] += time.perf_counter() - t
        self.det_pending = []

    def _resolve(self):
        """Interpole et neutralise les frames impaires dont les 2 voisines sont prêtes."""
        for pos in [p for p in self.held if (p - 1) in self.boxes_at and (p + 1) in self.boxes_at]:
            boxes = interpolate_boxes(self.boxes_at[pos - 1], self.boxes_at[pos + 1])
            idx, frame = self.held.pop(pos)
            neutralize_player(frame, boxes)
            self.emit_buf.append((idx, frame))
        #purge des boîtes devenues inutiles (plus aucune impaire voisine en attente)
        for p in [p for p in self.boxes_at
                  if p < self.position - 3 and (p + 1) not in self.held and (p - 1) not in self.held]:
            del self.boxes_at[p]

    def _flush(self, out_q, final=False):
        while len(self.emit_buf) >= self.batch or (final and self.emit_buf):
            chunk, self.emit_buf = self.emit_buf[:self.batch], self.emit_buf[self.batch:]
            out_q.put(([i for i, _ in chunk], [f for _, f in chunk]))

    def feed(self, idx, frame, out_q):
        pos = self.position
        self.position += 1
        if pos % self.every == 0:
            self.det_pending.append((pos, idx, frame))
            if len(self.det_pending) >= self.batch:
                self._detect()
        else:
            self.held[pos] = (idx, frame)  # interpolée plus tard
        self._resolve()
        self._flush(out_q)

    def finish(self, out_q):
        self._detect()
        self._resolve()
        #frames impaires restantes (fin de vidéo) : pas de voisine droite -> on prend ce qu'on a
        for pos in list(self.held):
            boxes = self.boxes_at.get(pos - 1, self.boxes_at.get(pos + 1, []))
            idx, frame = self.held.pop(pos)
            neutralize_player(frame, boxes)
            self.emit_buf.append((idx, frame))
        self._flush(out_q, final=True)


def _human_worker(model_human, detect_every, batch, in_q, out_q, timing):
    """Étage 2 : détection joueurs (1 frame sur `detect_every`) + interpolation."""
    stage = _HumanStage(model_human, batch, detect_every, timing)
    while True:
        item = in_q.get()
        if item is None:
            break
        idxs, frames = item
        for idx, frame in zip(idxs, frames):
            stage.feed(idx, frame, out_q)
    stage.finish(out_q)
    out_q.put(None)


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

    monitor = ResourceMonitor()
    monitor.start()
    with tqdm() as pbar:
        decode_t = threading.Thread(
            target=_decode_worker, args=(video, step, batch, live, raw_q, timing), daemon=True
        )
        human_t = threading.Thread(
            target=_human_worker,
            args=(model_human, HUMAN_DETECT_EVERY, batch, raw_q, neut_q, timing),
            daemon=True,
        )
        decode_t.start()
        human_t.start()
        #la balle (étage le plus coûteux) tourne dans le thread principal
        _ball_worker(model_ball, neut_q, data, pbar, timing)
        human_t.join()
        decode_t.join()
    monitor.stop_and_report()

    #temps cumulé par étage (wall-clock contendu : indicatif, pas du temps GPU pur)
    print(
        "temps cumulé par étage ⏱️  "
        f"decode: {timing['decode']:.1f}s | humain: {timing['human']:.1f}s | balle: {timing['ball']:.1f}s"
    )
    return data
