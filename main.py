from src.train import fit

if __name__ == "__main__":
    # Passe data_dir ggf. an deinen Pfad an:
    fit(
        data_dir="data/TrashType_Image_Dataset",
        models_dir="models",
        img_size=224,
        batch_size=96,
        epochs=35,
        lr=1e-3,
        val_split=0.2,
        use_class_weights=True,
        seed=42,
    )
