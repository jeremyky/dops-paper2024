import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set up AlexNet pre-trained model
alexnet = models.alexnet(pretrained=True)

# Set AlexNet to evaluation mode
alexnet.eval()

# Transformation for input images (resize, normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from a single image
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_preprocessed = preprocess(img)
    img_batch = img_preprocessed.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No gradient calculation needed
        features = alexnet(img_batch)

    return features

# Extract features from a directory of images
def extract_features_from_directory(image_directory):
    features_list = []
    for img_file in os.listdir(image_directory):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_directory, img_file)
            features = extract_features(img_path)
            features_list.append(features)

    return features_list

# Specify your image directory
image_directory = './images/'

# Extract features from the images in the directory
features = extract_features_from_directory(image_directory)

# Save the extracted features for future use
torch.save(features, 'extracted_features.pt')

# Display shape of extracted features (e.g., for the first image)
print(f"Extracted features shape for first image: {features[0].shape}")
