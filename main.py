import cv2
import numpy as np
import os
import time
import glob
from pathlib import Path
from person_profiles import main as launch_advanced_interface  # Import the advanced interface

# Download models if they don't exist
def download_models():
    # Paths for models
    prototxt_path = "deploy.prototxt"
    caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
    openface_path = "openface_nn4.small2.v1.t7"
    
    if not os.path.exists(prototxt_path):
        print("Downloading face detection model files...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
    
    if not os.path.exists(caffemodel_path):
        print("Downloading face detection caffemodel...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            caffemodel_path
        )
    
    if not os.path.exists(openface_path):
        print("Downloading face recognition model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7",
            openface_path
        )

# First, download the required models
print("Checking for required model files...")
download_models()

# Launch the advanced interface from person_profiles.py
if __name__ == "__main__":
    print("Launching L1GHT REC0N Advanced Face Recognition System...")
    launch_advanced_interface()