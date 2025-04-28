import cv2
import face_recognition
import numpy as np
import os
import time
import glob

# Function to load known faces from dataset directory
def load_known_faces(dataset_dir):
    known_face_encodings = []
    known_face_names = []
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist. Creating it now.")
        os.makedirs(dataset_dir)
        print(f"Please add face images to the dataset directory {dataset_dir}.")
        print("Folder structure should be: dataset/PersonName/image.jpg")
        
        # Check if there are any images in root directory we can use
        sample_images = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
        if sample_images:
            print(f"Found {len(sample_images)} images in root directory. Adding to dataset...")
            for img in sample_images[:5]:  # Limit to 5 images for now
                person_name = os.path.splitext(img)[0]
                person_dir = os.path.join(dataset_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)
                import shutil
                shutil.copy(img, os.path.join(person_dir, img))
                print(f"Added {img} to dataset as {person_name}")
    
    # Walk through all directories in the dataset folder
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            print(f"Loading images for {person_name}...")
            # Load each image file in the person's directory
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        # Load image and compute face encoding
                        image = face_recognition.load_image_file(img_path)
                        # Try to find a face in the image
                        face_locations = face_recognition.face_locations(image)
                        if face_locations:
                            encoding = face_recognition.face_encodings(image, face_locations)[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                            print(f"  Added {img_file}")
                        else:
                            print(f"  No face found in {img_file}, skipping")
                    except Exception as e:
                        print(f"  Error processing {img_file}: {e}")
    
    return known_face_encodings, known_face_names

# Load dataset
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
print(f"Loading face dataset from: {dataset_dir}")
known_face_encodings, known_face_names = load_known_faces(dataset_dir)

print(f"Loaded {len(known_face_encodings)} face(s) for {len(set(known_face_names))} person(s)")

# Initialize webcam with a higher resolution
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Variables for FPS calculation
prev_time = time.time()
frame_count = 0
fps = 0

# Set the scale factor for frame resizing (smaller values = faster processing but less accurate)
scale_factor = 0.25  # Process at 25% of original size

print("Press 'q' to quit")

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        
        # Match faces against known faces
        for face_encoding in face_encodings:
            # Compare with all known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # If we found a match, use the known face with the smallest distance
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
            face_names.append(name)
        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= int(1/scale_factor)
            right *= int(1/scale_factor)
            bottom *= int(1/scale_factor)
            left *= int(1/scale_factor)
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
