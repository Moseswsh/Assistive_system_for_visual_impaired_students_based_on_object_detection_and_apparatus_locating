# Assistive_system_for_visual_impaired_students_based_on_object_detection_and_apparatus_locating
In this project, an assistive system was provided to locate appatatus location on benches for visual-impaired students by Object detection module and Zone locating module based on YOLOv5 and HSV model


## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)

## Project Overview

This project uses the YOLO model to detect and identify predefined zones on a lab bench and the objects within these zones. Main functionalities include:

1. **Real-time Camera Connection**: Captures a live video stream from a connected camera.
2. **Zone Recognition**: Divides the lab bench into multiple zones with designated labels.
3. **Object Detection and Positioning**: Detects apparatus within specific zones and provides both visual and audio feedback.

## Installation

Install the required dependencies using `requirements.txt`:

    pip install -r requirements.txt

## Project Structure

The main files and folders in this project are organized as follows:

- `Realtime_recognition.py`: The main script that connects to the camera, recognizes zones on the bench, identifies apparatus, and determines their locations.
- `Datasets/`: Contains YOLO-format datasets used for model training.
- `Trained_models/`: Stores the pre-trained model weights used for object detection.

## Usage

1. **Prepare Datasets and Model**  
   Place YOLO-format training data into the `Datasets/` folder and ensure the trained model weights are stored in the `Trained_models/` folder.

2. **Run the Main Script**  
   Execute `Realtime_recognition.py` to start real-time recognition:

        python Realtime_recognition.py

3. **Real-time Detection and Feedback**  
   The program will connect to the camera, recognize the bench zones, and identify any apparatus in each zone. Results are displayed with text and visual markers, along with audio feedback.

## File Descriptions

- **Realtime_recognition.py**  
  The main script responsible for connecting to the camera, recognizing zone numbers, identifying apparatus within zones, and providing visual and audio feedback.

- **Datasets/**  
  Contains YOLO-format datasets for training the detection model. Place new training data here if retraining is needed.

- **Trained_models/**  
  Contains trained YOLO model weights. The main script loads these weights for real-time detection.

---

