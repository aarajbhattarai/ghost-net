import torch
import numpy as np

def compute_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

def verify_face(face1, face2, model, threshold=0.7):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Preprocess the face images (assuming they are already resized and normalized)
    face1 = torch.from_numpy(face1).unsqueeze(0).float()
    face2 = torch.from_numpy(face2).unsqueeze(0).float()
    
    with torch.no_grad():
        # Generate embeddings for both faces
        embedding1 = model(face1)
        embedding2 = model(face2)
    
    # Compute similarity between the embeddings
    similarity = compute_similarity(embedding1, embedding2)
    
    # Return True if the similarity is above the threshold, indicating a match
    return similarity > threshold
