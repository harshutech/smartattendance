# pip install opencv-python==4.5.2

# use a open cv2 to to collect photos from video 
# ask for ID


# import cv2 

# video=cv2.VideoCapture(0)

# # inbuld classifer CascadeClassifier to deteceh face 
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# id = input("Enter Your ID: ")
# # id = int(id)
# count=0

# while True:
#     ret,frame=video.read()
#     # means takes photo in black white mode 
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # allow takes multiphotos 
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x,y,w,h) in faces:
#         count=count+1
#         cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

#     cv2.imshow("Frame",frame)

#     k=cv2.waitKey(1)

#     if count>500:
#         break

# video.release()
# cv2.destroyAllWindows()
# print("Dataset Collection Done..................")


import cv2
import os

# Start video capture from the default webcam
video = cv2.VideoCapture(0)

# Load the pre-trained face detection classifier
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ask for the user ID
user_id = input("Enter Your ID: ")

# Create a directory for the datasets if it doesn't exist
user_folder = 'datasets'  # All images will be stored in this folder
os.makedirs(user_folder, exist_ok=True)

count = 0

while True:
    ret, frame = video.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        
        # Save the detected face in the datasets folder with a unique filename
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(user_folder, f'User.{user_id}.{count}.jpg'), face_img)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    
    # Exit the loop if the count reaches 500
    if count >= 500:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video.release()
cv2.destroyAllWindows()

print("Dataset Collection Done..................")
