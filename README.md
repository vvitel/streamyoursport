# streamyoursport

## Objectif

Distinguer les phases de jeu à partir d'un flux vidéo issu d'une caméra unique.

## Principe

On applique un modèle yolo sur les frames de la vidéo.<br>
Afin de réduire les détections de faux positifs, on efface les joueurs du terrain.<br><br>
Les détections avec moins de 0.7 de confiance et trop proches entre deux frames successives sont supprimées (distance euclidienne).<br>
L'objectif est de ne détecter que des balles en mouvement (i.e. pas dans le filet, pas dans la main d'un joueur...).

## Exécuter le code

Le code est optimisé pour un GPU NVIDIA (L4) : inférence **TensorRT FP16** (les deux
modèles YOLO) et décodage vidéo **NVDEC** via torchcodec, les frames étant traitées
par lots de 8.

Passez les poids PyTorch (`.pt`) : au premier lancement, un moteur TensorRT `.engine`
est exporté à côté du `.pt` puis réutilisé ensuite. L'export doit se faire sur la
machine GPU cible (un `.engine` est spécifique au GPU et à la version de TensorRT).

```bash
python -B ./main.py --model_path "./weights/best_padel_s_1.pt" --video_path "/chemin/vers/vid2.mp4" --frame_step 3
```