import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect_faces(self, image):
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = self.face_detection.process(image_rgb)
        
        if results.detections:
            return results.detections
        return []

    def draw_faces(self, image, detections):
        for detection in detections:
            # Draw the face detection box
            self.mp_drawing.draw_detection(image, detection)
        
        return image

    def __del__(self):
        self.face_detection.close()
