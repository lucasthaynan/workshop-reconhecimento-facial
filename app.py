from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

video_capture = cv2.VideoCapture("video/debate_2022.mp4")

lula_image = face_recognition.load_image_file("fotos_treino/lula.png")
lula_face_encoding = face_recognition.face_encodings(lula_image)[0]

bolsonaro_image = face_recognition.load_image_file("fotos_treino/bolsonaro.jpg")
bolsonaro_face_encoding = face_recognition.face_encodings(bolsonaro_image)[0]

known_face_encodings = [
    lula_face_encoding,
    bolsonaro_face_encoding
]

known_face_names = [
    "Lula",
    "Bolsonaro"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def detect_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_small_frame, model="CNN", number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.58)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def gen_frames():
    while True:
        success, frame = video_capture.read()

        if not success:
            break

        face_locations, face_names = detect_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            if name == "Desconhecido":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 165, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
