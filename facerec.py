import cv2
import numpy as np
from deepface import DeepFace
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Load the cascade
face_cascade = cv2.CascadeClassifier('/Users/shandilya/Desktop/haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Get the width and height of the camera feed
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the width of the panel (you can adjust this according to your preference)
panel_width = 200

# Define colors
white = (255, 255, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

# Initialize the recognized students dictionary with recognition counts
recognized_students = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Clear the recognized students dictionary for the current frame
    recognized_students_this_frame = {}

    # Perform face recognition for each detected face
    for (x, y, w, h) in faces:
        # Crop the detected face from the frame
        detected_face = frame[y:y+h, x:x+w]
        file_name = None
        # Perform face recognition
        try:
            recognition_result = DeepFace.find(detected_face, db_path="/Users/shandilya/Desktop/faces_db")
            file_path = recognition_result[0]['identity'].iloc[0]
            file_name = file_path.split("/")[-1].split(".")[0]
            recognized_students_this_frame[file_name] = (x, y, w, h)
            recognized_students.add(file_name)

            # Draw the name of the detected face on top of the bounding box
            cv2.putText(frame, file_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as ex:
            print(f"Exception caught: {ex}")

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Create a black panel on the right side of the feed
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)

    # Display the names of recognized students and their recognition count on the panel
    print(f"Recognized Students: {recognized_students}")
    idx=0
    for student_name in recognized_students:
        idx += 1
        cv2.putText(panel, f"{student_name}", (10, 30 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Concatenate the frame and the panel horizontally
    feed_with_panel = np.hstack((frame, panel))

    # Display the frame with the panel
    cv2.imshow('Frame with Panel', feed_with_panel)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
