import cv2
import os
import csv
from datetime import datetime

# Load the trained LBPHFaceRecognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# Load the face cascade classifier
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# List of names corresponding to IDs in the trained model
name_list = ["Unknown", "Harsh Patil", "Mukul Gupta"]  # Ensure this matches the trained IDs

# Set to track names that have already been marked present today
present_today = set()

# Function to update attendance in daily CSV file
def update_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    attendance_folder = "attendance_logs"
    
    # Ensure the attendance logs folder exists
    os.makedirs(attendance_folder, exist_ok=True)

    # File path for today's attendance
    filename = os.path.join(attendance_folder, f"{date}_attendance.csv")

    # Check if the file exists. If not, create it with headers.
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])  # Adding headers

    # Append the attendance record
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date, now.strftime("%H:%M:%S")])
    
    print(f"Attendance updated for {name} on {date}")

# Video capture from default webcam
video = cv2.VideoCapture(0)

# Main loop for face recognition and attendance updating
while True:
    ret, frame = video.read()
    if not ret:
        print("Error reading video feed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]

        # Perform recognition on the face ROI
        serial, confidence = recognizer.predict(face_roi)

        # Ensure serial is within the bounds of the name list
        if serial >= 0 and serial < len(name_list):
            recognized_name = name_list[serial]
        else:
            recognized_name = "Unknown"

        # Print recognition details for debugging
        print(f"Recognized: {recognized_name}, Confidence: {confidence}")

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # If the confidence is below a certain threshold and not already marked present
        if confidence < 70 and recognized_name != "Unknown" and recognized_name not in present_today:
            # Update attendance
            update_attendance(recognized_name)
            # Mark this user as present
            present_today.add(recognized_name)
            # Display name
            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif confidence >= 70:
            # If confidence is too low, label as Unknown
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
print("Attendance updated and program terminated.")






# import cv2
# import os
# import csv
# import numpy as np
# from datetime import datetime

# # Load the trained LBPHFaceRecognizer model
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("Trainer.yml")

# # Load the face cascade classifier
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # List of names corresponding to IDs in the trained model
# name_list = [" ", "Mridul", "Tiwari", "Harsh" ]  # Update with your actual names

# # Function to update attendance in CSV file
# def update_attendance(name, date_str):
#     filename = f"{date_str}_attdendance.csv"
#     with open(filename, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

# # Video capture from default webcam
# video = cv2.VideoCapture(0)

# # Main loop for face recognition and attendance updating
# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Error reading video feed")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # Keep track of users who have already been marked as present during this time window
#     present_users = {}

#     for (x, y, w, h) in faces:
#         # Crop face region of interest (ROI)
#         face_roi = gray[y:y+h, x:x+w]

#         # Perform recognition on the face ROI
#         serial, confidence = recognizer.predict(face_roi)

#         # Get the recognized name
#         recognized_name = name_list[serial]

#         # Print recognition details for debugging
#         print("Recognized:", recognized_name, "Confidence:", confidence)

#         # Draw rectangle around the face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # If the confidence is below a certain threshold, update attendance and display name
#         if confidence < 70:
#             now = datetime.now()
#             date_str = now.strftime("%Y-%m-%d")
#             time_str = now.strftime("%H:%M:%S")

#             # Check if the user has already been marked as present during this time window
#             if recognized_name not in present_users or present_users[recognized_name] != time_str:
#                 # Update attendance
#                 update_attendance(recognized_name, date_str)
#                 present_users[recognized_name] = time_str
#                 # Display name
#                 cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#             else:
#                 # If the user has already been marked as present, display "Already marked as present"
#                 cv2.putText(frame, "Already marked as present", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#         else:
#             # If confidence is too low, label as Unknown
#             cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     cv2.imshow("Frame", frame)

#     k = cv2.waitKey(1)
#     if k == ord("q"):
#         break

# # Release video capture and close windows
# video.release()
# cv2.destroyAllWindows()
# print("Attendance updated and program terminated.")