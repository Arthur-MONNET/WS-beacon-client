# Vulpo - Beacon

Ce dépôt contient le code source du client Beacon pour le projet Vulpo.

## Installation

Assurez-vous d'avoir Python 3.x installé sur votre système.

1. Clonez ce dépôt en utilisant la commande suivante :

```bash
git clone https://github.com/Arthur-MONNET/WS-beacon-client.git
```

2. Accédez au répertoire du projet :

```bash
cd WS-beacon-client
```

3.Exécutez le script `setup.sh` pour configurer l'environnement :
   
```bash
./setup.sh
```

## Utilisation

Exécutez le script `classify.py` pour démarrer la reconnaissance sonore :

sur un `Raspberry Pi` :

```bash
su pi -c '/usr/bin/python3 /home/pi/Documents/WS-beacon-client/classify.py --ws <url du serveur> --model <chemin absolu du modèle>
```

sur un `PC` :

```bash
python3 classify.py --ws <url du serveur> --model <chemin absolu du modèle>
```

## Configuration

Le fichier `classify.py` offre plusieurs options de configuration :
- `--ws` : l'URL du serveur WebSocket
- `--model` : le chemin absolu du modèle à utiliser (par défaut : `yamnet.tflite`)
- `--maxResult` : le nombre de résultats à afficher (par défaut : `5`)
- `--overlappingFactor` : le facteur de chevauchement entre les interférences adjacentes (compris entre `0` et `1`, par défaut : `0.0`)
- `--scoreThreshold` : Seuil de score pour les résultats (compris entre `0` et `1`, par défaut : `0.0`)
- `--numThreads` : le nombre de threads CPU à utiliser (par défaut : `4`)
- `--enableEdgeTPU` : Exécutez le modèle sur EdgeTPU (par défaut : `False`)
