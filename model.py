
import torch
from torch import nn
from torchvision import models, transforms


CLASS_NAMES = [
    "Sun-Damaged Skin Lesion",
    "Basal Cell Skin Cancer",
    "Non-Cancerous Skin Growth",
    "Usually Harmless Skin Bump",
    "Melanoma (Serious Skin Cancer)",
    "Common Mole (Pigmented Skin Spot)",
    "Blood Vessel-Related Skin Mark",
]


CLASS_CODES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

def create_model(pretrained=False):
    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if pretrained else None
    )

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features, len(CLASS_NAMES)
    )

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(model_path="models/skin_model.pth"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model(pretrained=False)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    if checkpoint["class_codes"] != CLASS_CODES:
        raise ValueError("Model class order mismatch")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict_image(image, model, device):
    transform = get_transform()

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    probabilities = probabilities.cpu().numpy()
    predicted_index = int(probabilities.argmax())

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(prob)
            for name, prob in zip(CLASS_NAMES, probabilities)
        }
    }
