# import cv2
# import os
# import numpy as np #LBPH uses numpy to record LBPH output 
# from PIL import Image

# # Path for face image database
# path = 'datasets'

# # makes histogram of 500 images 
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# def getImageID(path):
#     imagePath = [os.path.join(path, f) for f in os.listdir(path)]
#     faces = []
#     ids = []
#     for imagePaths in imagePath:
#         try:
#             faceImage = Image.open(imagePaths).convert('L')
#             faceNP = np.array(faceImage, 'uint8')  # Ensure data type is uint8
#             Id = int(os.path.split(imagePaths)[-1].split(".")[1])
#             faces.append(faceNP)
#             ids.append(Id)
#             cv2.imshow("Training", faceNP)
#             cv2.waitKey(1)
#         except Exception as e:
#             print(f"Error processing image {imagePaths}: {e}")
#             continue

#     return ids, faces

# # skip bkur images and hilnde dhulne wali photos 
# try:
#     IDs, facedata = getImageID(path)
#     recognizer.train(facedata, np.array(IDs))
#     recognizer.save("Trainer.yml")
#     cv2.destroyAllWindows()
#     print("Training Completed............")
# except Exception as e:
#     print(f"An error occurred during training: {e}")


import os
import cv2
import numpy as np
from PIL import Image
import stat

# Set the path to your datasets directory
datasets_dir = "datasets"

# Automatically get the list of folders in the datasets directory
folders = [os.path.join(datasets_dir, folder) for folder in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, folder))]

for folder in folders:
    try:
        os.chmod(folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print(f"Changed permissions for {folder}")
    except Exception as e:
        print(f"Error changing permissions for {folder}: {e}")


# Path for face image database and trained user files
path = 'datasets'
trained_users_dir = 'trained_users'

# Create the trained_users directory if it doesn't exist
if not os.path.exists(trained_users_dir):
    os.makedirs(trained_users_dir)

# Initialize LBPH recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get image paths, face data, and new untrained user IDs
def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    new_ids = []  # List of new untrained user IDs

    for imagePaths in imagePath:
        try:
            faceImage = Image.open(imagePaths).convert('L')
            faceNP = np.array(faceImage, 'uint8')
            Id = int(os.path.split(imagePaths)[-1].split(".")[1])

            # Check if the user is already trained
            trained_file_path = os.path.join(trained_users_dir, f"User_{Id}_trained.txt")
            if os.path.exists(trained_file_path):
                print(f"User {Id} is already trained. Skipping...")
                continue  # Skip if already trained

            # Collect face data and IDs for new users
            faces.append(faceNP)
            ids.append(Id)
            if Id not in new_ids:
                new_ids.append(Id)

            cv2.imshow("Training", faceNP)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Error processing image {imagePaths}: {e}")
            continue

    return ids, faces, new_ids

# Main training process
try:
    IDs, facedata, new_ids = getImageID(path)  # Retrieves user data
    if facedata:  # Only train if there are new faces
        recognizer.train(facedata, np.array(IDs))
        recognizer.save("Trainer.yml")

        # Mark the new users as trained
        for new_id in new_ids:
            with open(os.path.join(trained_users_dir, f"User_{new_id}_trained.txt"), 'w') as f:
                f.write(f"User {new_id} trained")

        cv2.destroyAllWindows()
        print("Training Completed............")
    else:
        print("No new users to train.")
except Exception as e:
    print(f"An error occurred during training: {e}")
