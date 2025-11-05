from src.train import fit

if __name__ == "__main__":
    fit(
        data_dir="data/TrashType_Image_Dataset", # wenn nötig, Pfad anpassen (wenn Bilder wo anders gespeichert wurden)
        models_dir="models",
        img_size=224,
        batch_size=96,
        epochs=5, # wenn Rechner ohne starker GPU verwendet wird -> geringe Anzahl an epochs
        lr=1e-3,
        val_split=0.2,
        use_class_weights=True,
        seed=42,
        preview=True
    )