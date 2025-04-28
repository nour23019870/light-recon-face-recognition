import cv2
import os
import time
import numpy as np
import json
from datetime import datetime
import random

# Check for GPU support
has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
if has_cuda:
    print(f"CUDA-enabled GPU detected: {cv2.cuda.getDevice()}")
else:
    print("No CUDA GPU detected. Running on CPU.")

# Load face detection model
print("Loading face detection model...")
face_detector_model = None

# Make sure model files exist
if not os.path.exists("deploy.prototxt") or not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
    print("Model files not found. Please run main_gpu.py first to download the models.")
    exit(1)

face_detector_model = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Enable GPU if available
if has_cuda:
    print("Setting DNN backend to CUDA")
    face_detector_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    face_detector_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def detect_faces(frame, confidence_threshold=0.5):
    """Detect faces in a frame using DNN model"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    
    face_detector_model.setInput(blob)
    detections = face_detector_model.forward()
    
    face_boxes = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding box is within frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            
            # Skip invalid boxes
            if startX >= endX or startY >= endY:
                continue
                
            # Format is (left, top, right, bottom) for OpenCV rectangle drawing
            face_boxes.append((startX, startY, endX, endY))
            
    return face_boxes

def collect_profile_info(person_name):
    """Collect profile information for a person"""
    print("\n" + "="*60)
    print(f"CREATING PROFILE FOR: {person_name.upper()}")
    print("="*60)
    print("Please provide the following information (or press Enter to use default/random values):")
    
    # Initialize with defaults
    profile_info = {
        "name": person_name,
        "age": None,
        "gender": None,
        "occupation": None,
        "nationality": None,
        "status": "CIVILIAN",
        "threat_level": "LOW",
        "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "",
        "sightings": 1
    }
    
    # Age
    age_input = input("Age: ")
    if age_input.strip():
        try:
            profile_info["age"] = int(age_input)
        except ValueError:
            print("Invalid age, using random value.")
            profile_info["age"] = random.randint(18, 65)
    else:
        profile_info["age"] = random.randint(18, 65)
    
    # Gender
    gender_input = input("Gender (Male/Female/Other): ").capitalize()
    if gender_input in ["Male", "Female", "Other"]:
        profile_info["gender"] = gender_input
    else:
        profile_info["gender"] = random.choice(["Male", "Female"])
    
    # Occupation
    occupation_input = input("Occupation: ")
    if occupation_input.strip():
        profile_info["occupation"] = occupation_input
    else:
        profile_info["occupation"] = random.choice(["Student", "Engineer", "Teacher", "Doctor", "Artist", "Programmer"])
    
    # Nationality
    nationality_input = input("Nationality: ")
    if nationality_input.strip():
        profile_info["nationality"] = nationality_input
    else:
        profile_info["nationality"] = random.choice(["USA", "Canada", "UK", "Germany", "Japan", "China", "India", "Australia"])
    
    # Status
    print("\nStatus Options: CIVILIAN, PERSON OF INTEREST, UNDER SURVEILLANCE, WANTED")
    status_input = input("Status: ").upper()
    if status_input in ["CIVILIAN", "PERSON OF INTEREST", "UNDER SURVEILLANCE", "WANTED"]:
        profile_info["status"] = status_input
    
    # Threat level
    print("\nThreat Level Options: LOW, MODERATE, HIGH")
    threat_input = input("Threat Level: ").upper()
    if threat_input in ["LOW", "MODERATE", "HIGH"]:
        profile_info["threat_level"] = threat_input
    
    # Notes
    notes_input = input("\nAdditional Notes: ")
    if notes_input.strip():
        profile_info["notes"] = notes_input
    else:
        profile_info["notes"] = "Profile created during face scanning session."
    
    print("\nProfile Information Summary:")
    for key, value in profile_info.items():
        if key != "last_seen":  # Skip timestamp in the summary
            print(f"- {key.capitalize()}: {value}")
    
    return profile_info

def save_profile_info(profile_info):
    """Save profile information to the person's dataset directory"""
    # Save profile file in the person's dataset directory
    person_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "dataset", profile_info["name"])
    os.makedirs(person_dir, exist_ok=True)
    
    personal_profile_path = os.path.join(person_dir, "profile.json")
    with open(personal_profile_path, 'w') as f:
        json.dump(profile_info, f, indent=4)
    
    print(f"Profile saved to {personal_profile_path}")

def main():
    # Get person's name
    person_name = input("Enter the person's name: ")
    if not person_name:
        print("No name provided. Exiting.")
        return
    
    # Collect profile information
    profile_info = collect_profile_info(person_name)
    
    # Save profile information
    save_profile_info(profile_info)
    
    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create person's directory if it doesn't exist
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Count existing files to continue numbering
    existing_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    start_index = len(existing_files) + 1
    
    print(f"\nFound {len(existing_files)} existing images for {person_name}")
    print(f"Saving new images to {person_dir}")
    
    # Ask for auto-capture option
    auto_capture = input("Enable auto-capture mode? (y/n): ").lower().startswith('y')
    if auto_capture:
        interval = float(input("Enter seconds between auto-captures (e.g., 3.0): "))
        print(f"Auto-capture enabled: Will capture an image every {interval} seconds when a face is detected")
    else:
        print("Press 'c' to capture an image")
    
    print("\nCONTROLS:")
    print("  - Press 'c' to manually capture an image")
    print("  - Press 'a' to toggle auto-capture mode")
    print("  - Press 'q' to quit")
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for FPS calculation
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    # Counter for captured images
    capture_count = 0
    target_count = int(input("How many images do you want to capture? "))
    
    # Countdown variables
    countdown_active = False
    countdown_start = 0
    countdown_duration = 3  # seconds
    
    # Auto-capture variables
    last_auto_capture_time = time.time()
    
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
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Detect faces in the frame
        face_boxes = detect_faces(frame)
        
        # Draw rectangles around detected faces
        for (left, top, right, bottom) in face_boxes:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Check for auto-capture
        if auto_capture and len(face_boxes) > 0 and (current_time - last_auto_capture_time) >= interval:
            if not countdown_active:  # Only start auto-capture if not already counting down
                countdown_active = True
                countdown_start = current_time
                print("Starting auto-capture countdown...")
        
        # Handle countdown
        if countdown_active:
            seconds_left = max(0, int(countdown_duration - (current_time - countdown_start)))
            if seconds_left > 0:
                # Show countdown
                cv2.putText(display_frame, f"Capturing in: {seconds_left}", 
                            (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # Save the frame if at least one face is detected
                if len(face_boxes) > 0:
                    # Get the largest face (assuming it's the main subject)
                    largest_face = max(face_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                    left, top, right, bottom = largest_face
                    
                    # Extract face with some margin (20%)
                    margin_x = int((right - left) * 0.2)
                    margin_y = int((bottom - top) * 0.2)
                    face_img = frame[
                        max(0, top - margin_y):min(frame.shape[0], bottom + margin_y),
                        max(0, left - margin_x):min(frame.shape[1], right + margin_x)
                    ]
                    
                    # Save the face image
                    image_path = os.path.join(person_dir, f"{person_name}_{start_index + capture_count}.jpg")
                    cv2.imwrite(image_path, face_img)
                    print(f"Saved {image_path}")
                    capture_count += 1
                    
                    # Update last auto-capture time
                    last_auto_capture_time = current_time
                    
                    if capture_count >= target_count:
                        print(f"Captured {capture_count} images. Finished!")
                        break
                else:
                    print("No face detected, skipping capture")
                
                countdown_active = False
        
        # Show the number of faces detected
        cv2.putText(display_frame, f"Detected: {len(face_boxes)} face(s)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show capture count
        cv2.putText(display_frame, f"Captured: {capture_count}/{target_count}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show person name and profile info
        cv2.putText(display_frame, f"Person: {person_name}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add key instructions at the bottom of the screen
        cv2.putText(display_frame, "Press 'c' to capture", 
                    (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show auto-capture status
        auto_status = "ON" if auto_capture else "OFF"
        cv2.putText(display_frame, f"Auto-capture: {auto_status}", 
                    (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if auto_capture else (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Face Scanner', display_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 'c' to start countdown for capture
        if key == ord('c') and not countdown_active:
            countdown_active = True
            countdown_start = current_time
            print("Countdown started...")
        
        # Press 'a' to toggle auto-capture mode
        if key == ord('a'):
            auto_capture = not auto_capture
            if auto_capture:
                print("Auto-capture mode enabled")
            else:
                print("Auto-capture mode disabled")
    
    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Update the profile with the number of images
    profile_info["images_count"] = capture_count
    save_profile_info(profile_info)
    
    print(f"\nFace scanning complete. {capture_count} images saved to {person_dir}")
    print("The profile has been created and saved.")
    print("\nYou can now run person_profiles.py to view the profile or main_gpu.py to use these images for face recognition")

if __name__ == "__main__":
    main()