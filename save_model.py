import torch
from src.model_cnn import SmallCNN
import json
from pathlib import Path

# Pfade
models_dir = Path("models")
weights_in = models_dir / "best_weights.pt"     # das, was beim Training erzeugt wurde
weights_out = models_dir / "final_model.pt"     # neuer Dateiname
classes_json = models_dir / "classes.json"

# Klassen laden (aus JSON-Datei)
with open(classes_json, "r", encoding="utf-8") as f:
    classes = json.load(f)

# Modell initialisieren
model = SmallCNN(num_classes=len(classes))
state_dict = torch.load(weights_in, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

# Modell speichern (komplett, nicht nur Gewichte)
torch.save({
    "model_state": model.state_dict(),
    "classes": classes,
}, weights_out)

print(f"✅ Modell gespeichert unter: {weights_out.resolve()}")
