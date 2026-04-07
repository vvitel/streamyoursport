# streamyoursport

## Objectif

Distinguer les phases de jeux à partir d'un flux vidéo issu d'une caméra unique.

## Principe

On applique un modèle yolo sur les frames de la vidéo.<br>
Afin de réduire les détections de faux positifs, on efface les joueurs du terrain.<br>
Les détections trop proches entre deux frames sont supprimées.<br><br>

## Exécuter le code
```bash
python -B .\main.py --model_path ".\weights\best_padel_s_1.pt" --video_path "C:/Users/Utilisateur/video/vid2.mp4" --background_path ".\backgrounds\om_vid2.png" --frame_step 3
```

