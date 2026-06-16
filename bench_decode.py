"""Diagnostic du décodage vidéo : NVDEC est-il réellement utilisé ?

Usage:
    python bench_decode.py /chemin/vers/video.mp4 [n_frames]

Compare le décodage séquentiel CUDA vs CPU et isole le coût de la copie/conversion.
Si "cuda" n'est pas nettement plus rapide que "cpu", NVDEC n'est pas câblé
(FFmpeg/torchcodec sans support GPU) -> c'est la cause du goulot d'étranglement.
"""
import sys
import time
import torch
import numpy as np
from torchcodec.decoders import VideoDecoder


def bench(video, device, n, seek_mode):
    #init (le scan d'index "exact" se fait ici, hors boucle)
    t0 = time.perf_counter()
    dec = VideoDecoder(video, device=device, seek_mode=seek_mode)
    init = time.perf_counter() - t0

    #décodage séquentiel pur (sans copie CPU), synchronisé pour un timing GPU correct
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for frame in dec:
        count += 1
        if count >= n:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    pure = time.perf_counter() - t0

    print(f"[{device:4s} | seek={seek_mode:11s}] init: {init:5.2f}s | "
          f"décode pur {count} frames: {pure:5.2f}s = {count / pure:6.1f} fps")
    return dec


def bench_with_copy(video, n):
    """Notre chemin réel : décode CUDA + permute + .cpu() + RGB->BGR."""
    dec = VideoDecoder(video, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for frame in dec:
        arr = frame.permute(1, 2, 0).cpu().numpy()[..., ::-1]
        arr = np.ascontiguousarray(arr)
        count += 1
        if count >= n:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[cuda + copie/BGR par frame      ] {count} frames: {dt:5.2f}s = {count / dt:6.1f} fps")


if __name__ == "__main__":
    video = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print(f"vidéo: {video} | torch CUDA dispo: {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun'}")
    print(f"metadata: {VideoDecoder(video).metadata}\n")

    bench(video, "cuda", n, "approximate")
    bench(video, "cuda", n, "exact")
    bench(video, "cpu", n, "approximate")
    bench_with_copy(video, n)
