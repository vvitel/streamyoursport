# streamyoursport

## Objectif

Distinguer les phases de jeu à partir d'un flux vidéo issu d'une caméra unique.

## Principe

On applique un modèle yolo sur les frames de la vidéo.<br>
La balle étant un objet difficile à détecter, il y a beaucoup de faux positifs.<br>
La plupart d’entre eux sont liés aux joueurs (raquettes, vêtements).<br>
Afin de les filtrer on ne garde que les détections en dehors des boxes des joueurs.<br>
Les détections avec une confiance trop faible et trop proches entre deux frames successives sont supprimées.<br>
L'objectif est de ne détecter que des balles en mouvement (i.e. pas dans le filet, pas dans la main d'un joueur...).<br>
## Exécuter le code
```bash
python -B .\main.py --video_path "C:/Users/Utilisateur/video/vid2.mp4" --frame_step 6
```

## Jeu de données d'entraînement 
<u>yolo balle de padel :</u> https://huggingface.co/datasets/Feculent/SYS
