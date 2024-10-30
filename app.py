import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import csv
from datetime import datetime
import time

# Initialize page settings
page_title = "FaceOlit"
page_icon = "WqegGTeNSs6upA9AgdhrQg.png"
logo_path = "WqegGTeNSs6upA9AgdhrQg.png"
datasets_dir = "datasets"
trained_users_dir = 'trained_users'

# Function to load attendance data from CSV file
def load_attendance_data(selected_date):
    filename = f"attendance_logs/{selected_date}_attendance.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df
    else:
        return None

# Function to update attendance in CSV file
def update_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"attendance_logs/{date_str}_attendance.csv"

    os.makedirs('attendance_logs', exist_ok=True)

    if os.path.exists(filename):
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0 and name == row[0]:
                    st.write(f"{name} already marked present.")
                    return

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        if os.stat(filename).st_size == 0:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, date_str, time_str])

    st.write(f"Attendance recorded for {name} at {time_str} on {date_str}.")

# Function to capture new user photos
def capture_user_photos(user_id, num_photos=50):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    count = 0

    user_folder = "datasets"
    os.makedirs(user_folder, exist_ok=True)

    st.write("Starting webcam for photo capture...")
    image_placeholder = st.empty()  # Placeholder for image

    while count < num_photos:
        ret, frame = video.read()
        if not ret:
            st.write("Failed to capture video feed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{user_folder}/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if count >= num_photos:
                st.write(f"Captured {count} images for user {user_id}.")
                break

        # Convert the frame to RGB and display it in Streamlit
        image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    video.release()
    cv2.destroyAllWindows()
    st.write(f"Image capture completed for User {user_id}.")

# Function to get image paths, face data, and new untrained user IDs
def get_image_id(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    new_ids = []

    for imagePaths in imagePath:
        try:
            faceImage = Image.open(imagePaths).convert('L')
            faceNP = np.array(faceImage, 'uint8')
            Id = int(os.path.split(imagePaths)[-1].split(".")[1])

            trained_file_path = os.path.join(trained_users_dir, f"User_{Id}_trained.txt")
            if os.path.exists(trained_file_path):
                st.write(f"User {Id} is already trained. Skipping...")
                continue

            faces.append(faceNP)
            ids.append(Id)
            if Id not in new_ids:
                new_ids.append(Id)

            cv2.imshow("Training", faceNP)
            cv2.waitKey(1)

        except Exception as e:
            st.write(f"Error processing image {imagePaths}: {e}")
            continue

    return ids, faces, new_ids

# Function to train the model
def train_model():
    path = "datasets"
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        IDs, face_data, new_ids = get_image_id(path)
        if face_data:
            recognizer.train(face_data, np.array(IDs))
            recognizer.save("Trainer.yml")

            for new_id in new_ids:
                with open(os.path.join(trained_users_dir, f"User_{new_id}_trained.txt"), 'w') as f:
                    f.write(f"User {new_id} trained")

            cv2.destroyAllWindows()
            st.success("Training Completed.")
        else:
            st.warning("No new users to train.")
    except Exception as e:
        st.error(f"An error occurred during training: {e}")

# Function to take attendance
def take_attendance():
    video = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    name_list = ["", "harsh patil", "mukul gupta"]

    os.makedirs("attendance_logs", exist_ok=True)

    st.write("Starting attendance process...")
    image_placeholder = st.empty()  # Placeholder for image

    while True:
        ret, frame = video.read()
        if not ret:
            st.write("Error reading video feed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            serial, confidence = recognizer.predict(face_roi)

            if confidence < 70:
                recognized_name = name_list[serial]
                update_attendance(recognized_name)
                st.write(f"Attendance recorded for {recognized_name}.")
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to RGB and display it in Streamlit
        image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    st.write("Attendance process completed.")

# Function to render the home page
def home():
    st.title("Welcome to FACEOLIT Smart Attendance Viewer")
    st.image("ClI7A1lWSLen7rFuVQBdCQ.jpg", width=500)
    st.write("This is a simple tool to view attendance records.")
    st.markdown("**FaceOlit** is a smart attendance management system utilizing facial recognition technology to track and manage attendance records efficiently.")

    if st.button("View Attendance"):
        attendance_view()

# Function to render the attendance viewing page
def attendance_view():
    st.title("Attendance Viewer")
    filenames = [filename for filename in os.listdir("attendance_logs") if filename.endswith("_attendance.csv")]
    dates_list = [filename.split('_')[0] for filename in filenames]

    selected_date = st.selectbox("Select Date", dates_list)

    df = load_attendance_data(selected_date)
    if df is not None:
        st.table(df.style.set_properties(align='center'))
    else:
        st.write(f"No attendance data available for {selected_date}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title=page_title, page_icon=page_icon)
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox("Select Page", ["Home", "Attendance Viewer", "Add New User", "Train Model", "Take Attendance"])

    if page == "Home":
        home()
    elif page == "Attendance Viewer":
        attendance_view()
    elif page == "Add New User":
        st.title("Add New User")
        user_id = st.text_input("Enter User ID:")
        if st.button("Capture Photos"):
            capture_user_photos(user_id)
    elif page == "Train Model":
        st.title("Train Model")
        if st.button("Train"):
            train_model()
    elif page == "Take Attendance":
        st.title("Take Attendance")
        if st.button("Start Attendance"):
            take_attendance()

if __name__ == "__main__":
    main()





# import streamlit as st
# import pandas as pd
# import os

# page_title = "FaceOlit"
# page_icon = "WqegGTeNSs6upA9AgdhrQg.png"
# logo_path = "WqegGTeNSs6upA9AgdhrQg.png"  # Path to your app's logo

# # Function to load attendance data from CSV file
# def load_attendance_data(selected_date):
#     filename = f"{selected_date}_attendance.csv"
#     if os.path.isfile(filename):
#         df = pd.read_csv(filename)
#         return df
#     else:
#         return None

# # Function to render the home page
# def home():
#     st.title("Welcome to FACEOLIT Smart Attendance Viewer")
#     st.image("ClI7A1lWSLen7rFuVQBdCQ.jpg", width=500)
#     st.write("This is a simple tool to view attendance records.")

#     # Add bold text about the web app
#     st.markdown("**FaceOlit** is a smart attendance management system utilizing facial recognition technology to track and manage attendance records efficiently.")

#     st.write("Please click the button below to view attendance.")
#     # Button to navigate to attendance viewing page
#     if st.button("View Attendance"):
#         attendance_view()


# # Function to render the attendance viewing page
# def attendance_view():
#     st.title("Attendance Viewer")

#     # Create a list of available dates from the filenames
#     filenames = [filename for filename in os.listdir() if filename.endswith("_attendance.csv")]
#     dates_list = [filename.split('_')[0] for filename in filenames]

#     # Select date from the list
#     selected_date = st.selectbox("Select Date", dates_list)

#     # Load attendance data based on selected date
#     df = load_attendance_data(selected_date)

#     if df is not None:
#         # Display attendance data in a table with better formatting
#         st.table(df.style.set_properties(align='center'))
#     else:
#         st.write(f"No attendance data available for {selected_date}")

# # Main function to run the Streamlit app
# def main():
#     st.set_page_config(page_title=page_title, page_icon=page_icon)
    
#     st.sidebar.image(logo_path, use_column_width=True)  # Display app logo
    
#     st.sidebar.markdown("---")
    
#     page = st.sidebar.selectbox("Select Page", ["Home", "Attendance Viewer"])
    

#     if page == "Home":
#         home()
#     elif page == "Attendance Viewer":
#         attendance_view()

# if __name__ == "__main__":
#     main()





# new 
# import streamlit as st
# import pandas as pd
# import os

# page_title = "FaceOlit"
# # page_icon = "WqegGTeNSs6upA9AgdhrQg.png"\
# page_icon = "WqegGTeNSs6upA9AgdhrQg.jpg"
# logo_path = "images.jpg"  # Path to your app's logo

# # Function to load attendance data from CSV file and ensure unique entries per name
# def load_attendance_data(selected_date):
#     filename = f"{selected_date}_attendance.csv"
#     if os.path.isfile(filename):
#         df = pd.read_csv(filename)

#         # Print the actual column names to help debug
#         st.write("Available columns:", df.columns.tolist())

#         # Check if 'Name' column exists in the CSV file
#         if 'Name' in df.columns:
#             # Remove duplicates based on 'Name' column, keeping the first occurrence
#             df = df.drop_duplicates(subset='Name', keep='first')
#         else:
#             st.error("'Name' column not found in the CSV file.")
#             return None
#         return df
#     else:
#         return None

# # Function to render the home page
# def home():
#     st.title("Presense Plus ")
#     st.image("ClI7A1lWSLen7rFuVQBdCQ.jpg", width=500)
#     st.write("This is a simple tool to view attendance records.")

#     # Add bold text about the web app
#     st.markdown("Presense Plus ")

#     st.write("Please click the button below to view attendance.")
#     # Button to navigate to attendance viewing page
#     if st.button("View Attendance"):
#         attendance_view()

# # Function to render the attendance viewing page
# def attendance_view():
#     st.title("Attendance Viewer")

#     # Create a list of available dates from the filenames
#     filenames = [filename for filename in os.listdir() if filename.endswith("_attendance.csv")]
#     dates_list = [filename.split('_')[0] for filename in filenames]

#     # Select date from the list
#     selected_date = st.selectbox("Select Date", dates_list)

#     # Load attendance data based on selected date
#     df = load_attendance_data(selected_date)

#     if df is not None:
#         # Display attendance data in a table with better formatting
#         st.table(df.style.set_properties(align='center'))
#     else:
#         st.write(f"No attendance data available for {selected_date}")

# # Main function to run the Streamlit app
# def main():
#     st.set_page_config(page_title=page_title, page_icon=page_icon)
    
#     st.sidebar.image(logo_path,  use_column_width=True)    # Display app logo
    
#     st.sidebar.markdown("---")
    
#     page = st.sidebar.selectbox("Select Page", ["Home", "Attendance Viewer"])

#     if page == "Home":
#         home()
#     elif page == "Attendance Viewer":
#         attendance_view()

# # This ensures the main function is called when the script is run directly
# if __name__ == "__main__":
#     main()