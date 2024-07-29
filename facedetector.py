import cv2
import numpy as np
from PIL import Image
import os
import sys
import io
import argparse
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import shutil
from retinaface import RetinaFace


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
DB_FOLDER = "DB"

def load_file(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return cv2.imread(file_path)
    elif file_path.lower().endswith(('.mp4', '.mkv')):
        return cv2.VideoCapture(file_path)
    else:
        raise ValueError("Unsupported file format")

def detect_faces(image, mode):
    if mode == 1:  # OpenCV Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [{'box': list(face)} for face in faces]
    elif mode == 2:  # YuNet
        height, width, _ = image.shape
        yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (width, height),
            0.9,  # score threshold
            0.3,  # nms threshold
            5000  # top k
        )
        _, faces = yunet_detector.detect(image)
        return [{'box': list(face[:4])} for face in faces] if faces is not None else []
    elif mode == 3:  # RetinaFace
        faces = RetinaFace.detect_faces(image)
        return [{'box': list(face['facial_area'])} for face in faces.values()]

def extract_face(image, face, margin=0.5):
    x, y, w, h = map(int, face['box'])
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(image.shape[1], x + w + margin_x)
    y_end = min(image.shape[0], y + h + margin_y)
    face_image = image[y_start:y_end, x_start:x_end]
    return face_image

def enhance_image(image, target_height=720, debug=False):
    if debug:
        print("Enhancing image...")
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    upscaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    if debug:
        print("Image enhancement complete.")
    return sharpened

def save_face(face_image, output_path, index, no_folder=False, debug=False):
    if debug:
        print(f"Saving face {index}...")
    if not no_folder:
        os.makedirs(output_path, exist_ok=True)
    enhanced = enhance_image(face_image, debug=debug)
    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    file_path = os.path.join(output_path, f"Face_{index}.png") if no_folder else f"{output_path}/Face_{index}.png"
    pil_image.save(file_path, quality=95, dpi=(300, 300))
    if debug:
        print(f"Face {index} saved.")
    return file_path

def save_full_frame(frame, face, output_path, index, no_folder=False, debug=False):
    if debug:
        print(f"Saving full frame for face {index}...")
    if not no_folder:
        os.makedirs(output_path, exist_ok=True)
    x, y, w, h = map(int, face['box'])
    frame_with_circle = frame.copy()
    
    center = (x + w//2, y + h//2)
    radius = int((w + h) / 3)
    cv2.circle(frame_with_circle, center, radius, (0, 255, 0), 3)
    
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.2, frame_with_circle, 0.8, 0, frame_with_circle)
    
    label = f"Face {index}"
    cv2.putText(frame_with_circle, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    file_path = os.path.join(output_path, f"Face_{index}_Frame.png") if no_folder else f"{output_path}/Face_{index}_Frame.png"
    cv2.imwrite(file_path, frame_with_circle, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if debug:
        print(f"Full frame for face {index} saved.")

def add_to_db(face_image_path):
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    shutil.copy(face_image_path, DB_FOLDER)

def process_frame(frame, mode, output_path, frame_index, face_count, max_faces, no_folder=False, debug=False):
    if debug:
        print(f"Processing frame {frame_index}...")
    faces = detect_faces(frame, mode)
    unique_faces = 0
    for i, face in enumerate(faces):
        if max_faces is not None and face_count['value'] >= max_faces:
            return unique_faces
        face_folder = output_path if no_folder else os.path.join(output_path, f"Face_{face_count['value'] + 1}")
        face_image = extract_face(frame, face)
        
        face_path = save_face(face_image, face_folder, face_count['value'] + 1, no_folder, debug)
        save_full_frame(frame, face, face_folder, face_count['value'] + 1, no_folder, debug)
        add_to_db(face_path)  # Add face to DB
        face_count['value'] += 1
        unique_faces += 1
    
    if debug:
        print(f"Frame {frame_index} processed. Found {len(faces)} faces, {unique_faces} unique.")
    return unique_faces

def parse_time(time_str):
    if '.' in time_str:
        t = datetime.strptime(time_str, "%M.%S")
        return timedelta(minutes=t.minute, seconds=t.second).total_seconds()
    else:
        return float(time_str)

def process_video(video_path, output_path, mode, time_range, max_faces, no_folder=False, debug=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")

    start_time, end_time = None, None
    if time_range:
        start_str, end_str = time_range.split('-')
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        total_frames = int((end_time - start_time) * fps)
    
    face_count = {'value': 0}
    total_unique_faces = 0
    
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        frame_index = 0
        while max_faces is None or face_count['value'] < max_faces:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end_time and current_time > end_time:
                break
            
            if frame_index % fps == 0:  # Process one frame per second
                unique_faces = process_frame(frame, mode, output_path, frame_index, face_count, max_faces, no_folder, debug)
                total_unique_faces += unique_faces
                if debug:
                    print(f"Frame {frame_index}: Detected {unique_faces} unique faces")
            
            frame_index += 1
            pbar.update(1)
    
    cap.release()
    return total_unique_faces, frame_index

def process_image(image_path, output_path, mode, max_faces, no_folder=False, debug=False):
    if debug:
        print("Processing image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return 0, 0
    faces = detect_faces(image, mode)
    faces_to_process = faces if max_faces is None else faces[:max_faces]
    
    face_count = {'value': 0}
    with tqdm(total=len(faces_to_process), desc="Processing", unit="face") as pbar:
        for i, face in enumerate(faces_to_process):
            face_folder = output_path if no_folder else os.path.join(output_path, f"Face_{face_count['value'] + 1}")
            face_image = extract_face(image, face)
            
            face_path = save_face(face_image, face_folder, face_count['value'] + 1, no_folder, debug)
            save_full_frame(image, face, face_folder, face_count['value'] + 1, no_folder, debug)
            add_to_db(face_path)  # Add face to DB
            face_count['value'] += 1
            
            pbar.update(1)
    
    if debug:
        print(f"Image processing complete. Found {face_count['value']} unique faces out of {len(faces_to_process)} detected.")
    return face_count['value'], 1

def main():
    parser = argparse.ArgumentParser(description="Advanced Face Detection and Recognition Application")
    parser.add_argument("-i", "--input", required=True, help="Path to input file (image or video)")
    parser.add_argument("-o", "--output", default="output", help="Path to output directory")
    parser.add_argument("-r", "--range", help="Time range for video processing (format: MM.SS-MM.SS or SS-SS)")
    parser.add_argument("-n", "--number", type=int, help="Maximum number of face outputs (default: unlimited)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed process information")
    parser.add_argument("-nf", "--no-folder", action="store_true", help="Save all outputs in the main output directory without creating separate folders for each face")
    parser.add_argument("-m", "--match", action="store_true", help="Match the input image to the database")
    parser.add_argument("-dupes", "--detect-duplicates", action="store_true", help="Detect and remove duplicate faces after processing")
    parser.add_argument("-mode", type=int, choices=[1, 2, 3], default=1, help="1: OpenCV, 2: YuNet, 3: RetinaFace (default: 1)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    start_time = time.time()

    print(f"Processing file: {args.input}")
    if args.input.lower().endswith(('.mp4', '.mkv')):
        total_faces, total_frames = process_video(args.input, args.output, args.mode, args.range, args.number, args.no_folder, args.debug)
    else:
        total_faces, total_frames = process_image(args.input, args.output, args.mode, args.number, args.no_folder, args.debug)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\nProcessing completed in {processing_time:.2f} seconds.")
    print(f"Detected {total_faces} unique faces in {total_frames} frames/images.")
    print(f"Output saved to {args.output}")
    print(f"Faces added to database in {DB_FOLDER}")

    if args.detect_duplicates:
        print("Detecting and removing duplicate faces...")
        # Implement duplicate detection logic here

if __name__ == "__main__":
    main()
