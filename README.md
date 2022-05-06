# Application_detection_visage

## Récupération des poids du model

Afin de pouvoir utilisé la reconnaissance de masque de l'application streamlit, il est tout d'abord nécessaire de récupérer le [fichier](https://drive.google.com/file/d/1s3pDdRV39ozqyFSf-3v80gnfQfK1YDa-/view?usp=sharing) *`.hdf5`* qui contient les poids du modèle de reconnaissance de masque. Le fichier est à placer dans le dossier **model**, ce qui donnera l'arborescence suivante :  

```
.
|__ cascades
|    |__ haarcascade_frontalface_default.xml
|
|__ images
|    |__ 004.jpg
|
|__ model
|    |__ model_weight.hdf5
|    |
|    |__ model.json
|
|__ app.py
|
|__ image_detect.py
|
|__ packages.txt
|
|__ README.md
|
|__ report.csv
|
|__ requirements.txt
```
__*Lien de récupération du fichier :*__ 
https://drive.google.com/file/d/1s3pDdRV39ozqyFSf-3v80gnfQfK1YDa-/view?usp=sharing

## Demarrage de l'application

Pour démarrer l'application il suffit de lancer la commande suivante :

```streamlit run app.py```