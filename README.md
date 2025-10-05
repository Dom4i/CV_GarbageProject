# Trash Type Image Classification
# Python-Version

Python 3.12.9

# Umgebung einrichten

Virtuelle Umgebung erstellen:
python -m venv .venv

Aktivieren:

Windows (PowerShell):
.venv\Scripts\Activate.ps1

macOS/Linux:
source .venv/bin/activate

# Abhängigkeiten installieren:
pip install -r requirements.txt

# Neue Abhängigkeiten hinzufügen

Wenn du ein neues Paket installierst, z. B.:
pip install opencv-python

Danach immer:
pip freeze > requirements.txt
Damit wird die aktuelle Paketliste für alle aktualisiert.

# Kaggle Dataset

Dataset herunterladen:
https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset?resource=download

Entpacken in den Ordner data

Die Ordnerstruktur sollte danach so aussehen:

data
└── Trash_Type_Image_Dataset
        ├── Cardboard
        ├── Glass
        ├── Metal
        ├── Paper
        ├── Plastic
        └── Trash


# Projekt starten

python main.py