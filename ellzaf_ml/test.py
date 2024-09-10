import cv2
from ellzaf_ml.models.face_detection import FaceDetector
import numpy as np
import torch

# Load an image
image = cv2.imread('./Aaraj.jpg')

# Create a FaceDetector instance
face_detector = FaceDetector()

# Detect faces in the image
faces = face_detector.detect_faces(image)

# Draw rectangles around the detected faces
image_with_faces = face_detector.draw_faces(image, faces)

# Display the image with detected faces
cv2.imshow('Face Detection', image_with_faces)
cv2.waitKey(0)
cv2.destroyAllWindows()
