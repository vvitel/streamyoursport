import os
from ultralytics import YOLO


def ensure_engine(model_path, batch=8, imgsz=None, half=True):
    """Retourne le chemin d'un moteur TensorRT (.engine), en l'exportant si besoin.

    - Si `model_path` est deja un .engine : on le renvoie tel quel.
    - Si c'est un .pt : on exporte un moteur TensorRT FP16 (half=True) a cote
      du .pt, avec un batch dynamique (1..batch). L'export n'a lieu qu'une fois,
      le .engine est ensuite reutilise.

    `imgsz` : taille d'entree du modele (cote du carre apres letterbox). NE PAS
    confondre avec la resolution video : chaque frame est redimensionnee a cette
    taille avant l'inference. Si None, on lit la taille d'entrainement des poids
    (le modele balle a ete entraine en 1280 pour bien voir la petite balle, le
    modele humain en 640) -- c'est le choix correct pour ne pas degrader la
    detection.

    L'export DOIT etre lance sur la machine GPU cible (L4) : un moteur TensorRT
    est specifique au GPU et a la version de TensorRT qui l'ont construit.
    """
    if model_path.endswith(".engine"):
        return model_path

    if not model_path.endswith(".pt"):
        raise ValueError(
            f"Modele attendu en .pt ou .engine, recu : {model_path}. "
            "Passez les poids PyTorch (ex: ./weights/best_padel_s_1.pt) pour "
            "qu'un moteur TensorRT puisse etre construit."
        )

    engine_path = model_path[:-3] + ".engine"
    if os.path.exists(engine_path):
        return engine_path

    model = YOLO(model_path)

    #si non precise, reprendre la taille d'entrainement des poids (ex: 1280 balle, 640 humain)
    if imgsz is None:
        trained = getattr(model.model, "args", {}).get("imgsz", 640)
        imgsz = trained[0] if isinstance(trained, (list, tuple)) else trained

    print(f"export TensorRT FP16 imgsz={imgsz} (batch dynamique <= {batch}) : {model_path} \U0001F527")
    exported = model.export(
        format="engine",
        half=half,        # FP16
        dynamic=True,     # batch variable 1..batch (gere le dernier lot partiel)
        batch=batch,      # batch maximal
        imgsz=imgsz,
        device=0,
    )
    return str(exported)
