import cv2  # biblioteca OpenCV (cv2) para manipulação e processamento de imagens e vídeos.
import face_recognition  # biblioteca para reconhecimento facial.
import numpy as np  # biblioteca NumPy para realizar operações matemáticas em arrays multidimensionais.
import pandas as pd # biblioteca para trabalhar com dados em diversos formatos, como CSV.
from tqdm import tqdm

# carregando vídeo a ser processado
video_capture = cv2.VideoCapture(f"video/debate_2022.mp4")

# carregando imagem do Lula de treino
lula_image = face_recognition.load_image_file(f"fotos_treino/lula.png")
lula_face_encoding = face_recognition.face_encodings(lula_image)[0]

# carregando imagem do Bolsonaro de treino
bolsonaro_image = face_recognition.load_image_file(f"fotos_treino/bolsonaro.jpg")
bolsonaro_face_encoding = face_recognition.face_encodings(bolsonaro_image)[0]

# salvando encondings das imagens em uma lista
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

df_dados = pd.DataFrame()

frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=frame_count, desc="Progresso do vídeo...")

# funcao para detectar as faces presentes no vídeos
def detecta_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="CNN", number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


time_video = 0
time_lula = 0
time_bolsonaro = 0

def processa_frames():

    # definindo variáveis de tempo como global
    global time_video
    global time_lula
    global time_bolsonaro

    while True:
        # tempo para cada frame do vídeo (0.0333 segundos)
        # site para ver o total de frames/s: https://www.flexclip.com/index.php?option=com_tools&lang=pt&view=metadata/
        time_video += 0.0333

        # Lê o próximo quadro do vídeo
        success, frame = video_capture.read()

        if not success:
            break

        # Detecta as faces no quadro e obtém as localizações e nomes
        face_locations, face_names = detecta_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4  # Redimensiona as coordenadas das faces para o tamanho original
            right *= 4
            bottom *= 4
            left *= 4

            # Desenha retângulos nas faces detectadas com base no nome
            if name == "Desconhecido":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Desenha retângulo vermelho na face desconhecida
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  # Desenha retângulo vermelho para o texto
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.85, (255, 255, 255), 1)  # Escreve o nome "Desconhecido" em branco

            else:
                if name.startswith('Lula'):
                    time_lula += 0.0333

                if name.startswith('Bolsonaro'):
                    time_bolsonaro += 0.0333

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)  # Desenha retângulo azul na face conhecida
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 165, 0), cv2.FILLED)  # Desenha retângulo azul para o texto
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  # Escreve o nome da face conhecida em branco

        pbar.update(1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    dados = {
        'tempo_video': round(time_video, 3),
        'tempo_pessoa': {
            'Lula': round(time_lula, 3),
            'Bolsonaro': round(time_bolsonaro, 3),
        },
        'percentual_pessoa': {
            'Lula': round((time_lula/time_video)*100, 2),
            'Bolsonaro': round((time_bolsonaro/time_video)*100, 2),

        }
    }

    df = pd.DataFrame(dados)

    print(df)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    pbar.close()


# executando a funcao
processa_frames()