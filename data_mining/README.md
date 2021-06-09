# [DOC] Data mining

# Création du dataset de pré-entraînement

## Pré-requis

- pandas version ≥ 1.2.4
- pytube version ≥ 10.8.4
- youtube-dl (cli tool)
- youtube_dl version ≥ 2021.5.16 (wrapper python pour youtube-dl)
- inaSpeechSegmenter version ≥ 0.6.7
- webrtcvad version ≥ 2.0.10
- tensorflow version ≥2.5.0

### Création d'une base de données de vidéos à partir de sources YouTube

```bash
usage: populate_bdd.py [-h] --source SOURCE --bdd BDD

optional arguments:
  -h, --help       show this help message and exit
  --source SOURCE  Fichier texte contenant les urls des playlists ou chaînes
                   YT
  --bdd BDD        Chemin vers le dossier contenant les BDD (pickle)
```

- Créé (ou importe si existants) les fichiers playlists.pkl et videos.pkl qui sont des bases de données sous forme de DataFrame stockés au format pickle.
- Dans un premier temps on construit une bdd de playlists à partir d'un fichier source contenant des urls de playlists.
- Ensuite on consruit une bdd de vidéos à partir de la bdd de playlists.
- On utilise pour ces deux étapes l'API youtube via la librairie pytube
- On utilise ces BDD pour l'étape de téléchargement via un autre script, et pour analyser les sources choisies sans avoir à télécharger les vidéos (stats sur la durée par exemple)

Script potentiellement instable car l'API YouTube est changeante et certaines vidéos peuvent disparaître d'une playlist si l'auteur la retire etc.

### Téléchargement et conversion des vidéos YouTube

```bash
usage: downloader.py [-h] --out OUT --bdd BDD

optional arguments:
  -h, --help  show this help message and exit
  --out OUT   Chemin de sauvegarde des fichiers audios
  --bdd BDD   Chemin vers le dossier contenant les BDD (pickle)
```

- Importe le DataFrame / base de données `videos.pkl` ****présent dans le dossier `bdd/`
- Pour chaque entrée :
    - Télécharge la vidéo dans le dossier `clips/`
    - Extrait l'audio et convertit au format 16 kHz 16bits mono wav
    - Met à jour la bdd (suivi du téléchargement, conversion, erreurs)

Script potentiellement instable car l'API YouTube est changeante et certaines vidéos peuvent disparaître d'une playlist si l'auteur la retire etc.

### Tri des sources entre fiable

Il n'y a pas de script pour cette partie, il faut le faire "à la main" en fonction des sources choisies au départ en entrant 1 ou 0 dans la colonne `fiable` du DataFrame `videos.pkl`.

Par défaut : toutes les sources sont fiables.

### Segmentation des audios pour ne garder que la voix

```bash
python segmenter.py -h

usage: segmenter.py [-h] --clips CLIPS --out OUT --bdd BDD

optional arguments:
  -h, --help     show this help message and exit
  --clips CLIPS  Chemin de sauvegarde des fichiers audios
  --out OUT      Chemin de sauvegarde des segments
  --bdd BDD      Chemin vers le dossier bdd
```

- Importe le DataFrame / base de données `videos.pkl` présent dans le dossier `bdd/`
- Pour la partie fiable des données, utilise [WebRTC-VAD](https://github.com/wiseman/py-webrtcvad) pour supprimer les silences et créer des segments audios d'environ 30 secondes
- Pour la partie non fiable des données, utilise [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) pour isoler la voix de la musique et des silences puis créée des semgents audios d'environ 30 secondes

## *A prévoir*

- *Seconde passe sur les fichiers toujours trop gros (> 45 sec) via inaSpeechSegmenter*
- *Systeme de suivi de la segmentation dans la bdd*