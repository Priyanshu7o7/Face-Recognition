import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Set the frame rate to 10 frames per second
video_capture.set(cv2.CAP_PROP_FPS, 10)

# Set the resolution to a quarter of the original resolution
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load known images and encodings
rohit_image = face_recognition.load_image_file("photos/rohitsharma.jpg")
rohit_encoding = face_recognition.face_encodings(rohit_image)[0]

kohli_image = face_recognition.load_image_file("photos/viratkohli.jpg")
kohli_encoding = face_recognition.face_encodings(kohli_image)[0]

gill_image = face_recognition.load_image_file("photos/shubhmangill.jpg")
gill_encoding = face_recognition.face_encodings(gill_image)[0]

kishan_image = face_recognition.load_image_file("photos/ishankishan.jpg")
kishan_encoding = face_recognition.face_encodings(kishan_image)[0]

rahul_image = face_recognition.load_image_file("photos/klrahul.jpg")
rahul_encoding = face_recognition.face_encodings(rahul_image)[0]

priyanshu_image = face_recognition.load_image_file("photos/priyanshushankar.jpg")
priyanshu_encoding = face_recognition.face_encodings(priyanshu_image)[0]

piyush_image = face_recognition.load_image_file("photos/piyushverma.jpg")
piyush_encoding = face_recognition.face_encodings(piyush_image)[0]

abhishek_image = face_recognition.load_image_file("photos/abhishekpuhan.jpg")
abhishek_encoding = face_recognition.face_encodings(abhishek_image)[0]

kartikey_image = face_recognition.load_image_file("photos/kartikey raj.jpg")
kartikey_encoding = face_recognition.face_encodings(kartikey_image)[0]

ashish_image = face_recognition.load_image_file("photos/ashishkumar.jpg")
ashish_encoding = face_recognition.face_encodings(ashish_image)[0]

# List of known face encodings and names
known_face_encodings = [
    rohit_encoding,
    kohli_encoding,
    gill_encoding,
    kishan_encoding,
    rahul_encoding,
    priyanshu_encoding,
    piyush_encoding,
    abhishek_encoding,
    kartikey_encoding,
    ashish_encoding
]

known_face_names = [
    "Rohit Sharma",
    "Virat Kohli",
    "Shubman Gill",
    "Ishan Kishan",
    "Kl Rahul",
    "Priyanshu Shankar",
    "Piyush Verma",
    "Abhishek Puhan",
    "Kartikey Raj",
    "Ashish Kumar"
]

students = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
                    print(f"Recognized: {name}")

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Attendance System made by Priyanshu Shankar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
