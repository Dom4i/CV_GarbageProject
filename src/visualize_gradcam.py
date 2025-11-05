def visualize_gradcam(model, img_path, class_names, target_layer="net.12"):
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # === Device wählen ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # === Bild laden und vorbereiten ===
    img = Image.open(img_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    x = tfm(img).unsqueeze(0).to(device)

    fmap = []
    grads = []

    # === Hooks für Forward + Backward ===
    def forward_hook(module, input, output):
        fmap.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    # === Gewählten Layer holen und Hooks registrieren ===
    layer = dict([*model.named_modules()])[target_layer]
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)  # modernere Variante

    # === Forward + Backward Pass ===
    logits = model(x)
    pred_class = logits.argmax(1).item()
    score = logits[0, pred_class]
    model.zero_grad()
    score.backward()

    # === Grad-CAM Berechnung ===
    grad_mean = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = (fmap[0] * grad_mean).sum(dim=1).squeeze().relu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cam.detach().cpu().numpy()

    # === Originalbild + Overlay anzeigen ===
    img_np = np.array(img.resize((224, 224))) / 255.0
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title(f"Grad-CAM for class: {class_names[pred_class]}")
    plt.axis("off")
    plt.show()
