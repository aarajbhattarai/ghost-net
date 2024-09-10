import torch
import numpy as np

def compute_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def verify_face(face1, face2, model, threshold=0.5):
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert images to tensors and add batch dimension
    face1_tensor = torch.from_numpy(face1).permute(2, 0, 1).float().unsqueeze(0)
    face2_tensor = torch.from_numpy(face2).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        embedding1 = model(face1_tensor)
        embedding2 = model(face2_tensor)

    similarity = compute_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())

    return similarity > threshold
