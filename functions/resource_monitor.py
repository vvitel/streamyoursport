import threading


class ResourceMonitor(threading.Thread):
    """Échantillonne périodiquement GPU / VRAM / NVDEC / CPU pendant un run.

    Sert à voir le vrai goulot d'étranglement (calcul GPU, mémoire, décodeur
    NVDEC ou CPU) et à juger s'il reste de la marge pour lancer plusieurs
    pipelines en parallèle. Dépendances optionnelles : pynvml (nvidia-ml-py)
    pour le GPU, psutil pour le CPU — absentes, on log seulement ce qu'on peut.
    """

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples = []
        self._nvml = None
        self._psutil = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

        try:
            import psutil
            self._psutil = psutil
            self._psutil.cpu_percent()  # amorce la mesure
        except Exception:
            self._psutil = None

    def run(self):
        while not self._stop_event.is_set():
            s = {}
            if self._nvml is not None:
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    dec = self._nvml.nvmlDeviceGetDecoderUtilization(self._handle)[0]
                    s["gpu"] = u.gpu
                    s["mem_used"] = m.used / 1e9
                    s["mem_total"] = m.total / 1e9
                    s["nvdec"] = dec
                except Exception:
                    pass
            if self._psutil is not None:
                try:
                    s["cpu"] = self._psutil.cpu_percent()
                except Exception:
                    pass
            if s:
                self.samples.append(s)
            self._stop_event.wait(self.interval)

    def stop_and_report(self):
        self._stop_event.set()
        self.join(timeout=2)

        if not self.samples:
            print("ressources 📊  (pynvml/psutil absents — pip install nvidia-ml-py psutil)")
            return

        def stat(key, fn):
            vals = [s[key] for s in self.samples if key in s]
            return fn(vals) if vals else 0.0

        avg = lambda k: stat(k, lambda v: sum(v) / len(v))
        peak = lambda k: stat(k, max)

        gpu_avg, gpu_peak = avg("gpu"), peak("gpu")
        mem_peak, mem_total = peak("mem_used"), peak("mem_total")
        nvdec_avg = avg("nvdec")
        cpu_avg = avg("cpu")

        print("ressources 📊")
        print(f"  GPU calcul : moy {gpu_avg:4.0f}% | pic {gpu_peak:4.0f}%")
        print(f"  VRAM       : pic {mem_peak:4.1f} / {mem_total:4.1f} Go")
        print(f"  NVDEC      : moy {nvdec_avg:4.0f}%")
        print(f"  CPU        : moy {cpu_avg:4.0f}%")

        #verdict : marge pour des pipelines en parallèle ?
        if gpu_avg >= 85:
            verdict = "calcul GPU saturé → peu de marge pour un 2e pipeline"
        elif cpu_avg >= 85:
            verdict = "CPU saturé → le goulot est côté CPU (pré/post-traitement)"
        elif mem_total and mem_peak > 0.7 * mem_total:
            verdict = "VRAM presque pleine → marge limitée par la mémoire"
        else:
            free = int(85 // max(gpu_avg, 1))
            verdict = (f"GPU à {gpu_avg:.0f}% et VRAM ok → de la marge : "
                       f"~{free} pipelines simultanés envisageables (à valider)")
        print(f"  verdict    : {verdict}")
