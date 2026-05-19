# streamyoursport

## Objectif

Distinguer les phases de jeu à partir d'un flux vidéo issu d'une caméra unique.

## Principe

On applique un modèle yolo sur les frames de la vidéo.<br>
Afin de réduire les détections de faux positifs, on efface les joueurs du terrain.<br><br>
Les détections avec moins de 0.7 de confiance et trop proches entre deux frames successives sont supprimées (distance euclidienne).<br>
L'objectif est de ne détecter que des balles en mouvement (i.e. pas dans le filet, pas dans la main d'un joueur...).

## Exécuter le code
```bash
python -B .\main.py --model_path "./weights/padel_openvino_model/" --video_path "C:/Users/Utilisateur/video/vid2.mp4" --frame_step 3
```