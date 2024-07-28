import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace

DB_FOLDER = "DB"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

def get_face_encoding(image_path, mode):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None

    height, width, _ = image.shape
    
    if mode == 1:  # OpenCV Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(opencv_faces) > 0:
            x, y, w, h = opencv_faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            return face_roi.flatten()
    
    elif mode == 2:  # YuNet
        yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (width, height),
            0.9,  # score threshold
            0.3,  # nms threshold
            5000  # top k
        )
        _, yunet_faces = yunet_detector.detect(image)
        if yunet_faces is not None and len(yunet_faces) > 0:
            yunet_face = yunet_faces[0]
            x, y, w, h = map(int, yunet_face[:4])
            yunet_face_roi = image[y:y+h, x:x+w]
            yunet_face_roi = cv2.resize(yunet_face_roi, (100, 100))
            yunet_face_roi_gray = cv2.cvtColor(yunet_face_roi, cv2.COLOR_BGR2GRAY)
            return yunet_face_roi_gray.flatten()
    
    elif mode == 3:  # RetinaFace
        retina_faces = RetinaFace.detect_faces(image)
        if retina_faces:
            retina_face = list(retina_faces.values())[0]
            x1, y1, x2, y2 = map(int, retina_face['facial_area'])
            retina_face_roi = image[y1:y2, x1:x2]
            retina_face_roi = cv2.resize(retina_face_roi, (100, 100))
            retina_face_roi_gray = cv2.cvtColor(retina_face_roi, cv2.COLOR_BGR2GRAY)
            return retina_face_roi_gray.flatten()
    
    return None

def compare_faces(encoding1, encoding2, threshold=0.8):
    if encoding1 is None or encoding2 is None:
        return False
    correlation = cv2.compareHist(
        np.histogram(encoding1, 256, [0, 256])[0].astype(np.float32),
        np.histogram(encoding2, 256, [0, 256])[0].astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    return correlation > threshold

def detect_duplicates(output_folder, mode):
    face_files = sorted([f for f in os.listdir(output_folder) if f.startswith("Face_") and f.endswith(".png") and not f.endswith("_Frame.png")])
    
    encodings = {}
    duplicates = []
    unique_faces = []
    
    print("Calculating face encodings...")
    for face_file in tqdm(face_files):
        face_path = os.path.join(output_folder, face_file)
        encoding = get_face_encoding(face_path, mode)
        encodings[face_file] = encoding
    
    print("Detecting duplicates...")
    for i, face_file in enumerate(tqdm(face_files)):
        if face_file in duplicates:
            continue
        
        current_encoding = encodings[face_file]
        is_unique = True
        
        for other_file in face_files[:i]:
            if other_file in duplicates:
                continue
            
            other_encoding = encodings[other_file]
            
            if compare_faces(current_encoding, other_encoding):
                duplicates.append(face_file)
                is_unique = False
                break
        
        if is_unique:
            unique_faces.append(face_file)
    
    print(f"Found {len(duplicates)} duplicate faces.")
    
    # Remove duplicate faces and all frame images except those corresponding to unique faces
    all_files = os.listdir(output_folder)
    for file in all_files:
        if file.startswith("Face_"):
            if file.endswith("_Frame.png"):
                if file.replace("_Frame.png", ".png") not in unique_faces:
                    os.remove(os.path.join(output_folder, file))
            elif file not in unique_faces:
                os.remove(os.path.join(output_folder, file))
    
    print(f"Removed {len(duplicates)} duplicate faces and their corresponding frames.")
    print(f"Kept {len(unique_faces)} unique faces and their frames.")

    # List remaining files
    remaining_files = sorted([f for f in os.listdir(output_folder) if f.startswith("Face_")])
    print("Remaining files:")
    for file in remaining_files:
        print(file)

def match_face(image_path, mode):
    query_encoding = get_face_encoding(image_path, mode)
    if query_encoding is None:
        print(f"No face detected in the query image: {image_path}")
        return []

    matches = []
    for db_image in os.listdir(DB_FOLDER):
        db_image_path = os.path.join(DB_FOLDER, db_image)
        db_encoding = get_face_encoding(db_image_path, mode)
        if db_encoding is not None:
            similarity = cv2.compareHist(
                np.histogram(query_encoding, 256, [0, 256])[0].astype(np.float32),
                np.histogram(db_encoding, 256, [0, 256])[0].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            if similarity > 0.8:  # Adjust this threshold as needed
                matches.append((db_image_path, similarity))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def main():
    parser = argparse.ArgumentParser(description="Duplicate Face Detection and Matching using multiple models")
    parser.add_argument("-d", "--detect", help="Path to the output folder for duplicate detection")
    parser.add_argument("-m", "--match", help="Path to the image for face matching")
    parser.add_argument("-mode", type=int, choices=[1, 2, 3], default=1, help="1: OpenCV, 2: YuNet, 3: RetinaFace (default: 1)")
    args = parser.parse_args()

    if args.detect:
        detect_duplicates(args.detect, args.mode)
    elif args.match:
        matches = match_face(args.match, args.mode)
        print(f"\n{len(matches)} matches found.")
        for match, similarity in matches:
            print(f"Match found: {match}, similarity: {similarity:.4f}")
    else:
        print("Please specify either -d for duplicate detection or -m for face matching.")

if __name__ == "__main__":
    main()