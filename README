# Reconhecimento Facial em Vídeos

Este é um código em Python para reconhecimento facial em vídeos utilizando a biblioteca OpenCV e face_recognition.

## Pré-requisitos

- Python 3.x

## Instalação das bibliotecas

Você pode instalar as bibliotecas necessárias executando o seguinte comando no terminal:

```shell
pip install -r requirements.txt

```

Certifique-se de ter o arquivo requirements.txt com as seguintes linhas:

```
opencv-python==4.7.0.72
opencv-python-headless==4.7.0.72
opencv_contrib_python==4.7.0.72
flask==2.3.2
face-recognition==1.3.0
tqdm==4.65.0

```

Isso irá instalar as versões específicas das bibliotecas necessárias para executar o código.

### Como usar

1. Certifique-se de ter um vídeo no formato MP4 disponível para processamento. Você pode alterar o caminho do vídeo na linha video_capture = cv2.VideoCapture(f"video/debate_2022.mp4") para o local do seu vídeo.

2. Faça o download das imagens de treinamento para cada pessoa que você deseja reconhecer. Neste código de exemplo, temos as imagens do Lula e do Bolsonaro. Certifique-se de ter as imagens corretas e atualize os caminhos das imagens de treinamento na seção de carregamento de imagens, como mostrado abaixo:

# carregando imagem do Lula de treino
lula_image = face_recognition.load_image_file(f"fotos_treino/lula.png")
lula_face_encoding = face_recognition.face_encodings(lula_image)[0]

# carregando imagem do Bolsonaro de treino
bolsonaro_image = face_recognition.load_image_file(f"fotos_treino/bolsonaro.jpg")
bolsonaro_face_encoding = face_recognition.face_encodings(bolsonaro_image)[0]

Certifique-se de que as imagens de treinamento estejam no formato correto (PNG ou JPEG) e correspondam à pessoa que você deseja reconhecer.

3. Execute o código. Ele processará o vídeo, detectará as faces presentes e as comparará com as imagens de treinamento para reconhecimento facial.

4. O resultado será exibido em uma janela de vídeo, onde retângulos serão desenhados nas faces detectadas e os nomes correspondentes serão exibidos acima de cada retângulo. As faces desconhecidas serão marcadas como "Desconhecido".

5. Para interromper a execução do código, pressione a tecla 'q' no teclado.

6. Após a conclusão do processamento do vídeo, será exibido um DataFrame com informações sobre o tempo total do vídeo, o tempo de cada pessoa reconhecida (Lula e Bolsonaro) e o percentual de tempo que cada pessoa apareceu no vídeo.

Certifique-se de que as imagens de treinamento e o vídeo estejam no formato correto e siga as etapas acima para utilizar o código corretamente.

# Explicação do código

O código utiliza a biblioteca OpenCV (cv2) para manipulação e processamento de imagens e vídeos, e a biblioteca face_recognition para o reconhecimento facial. Ele também faz uso das bibliotecas NumPy (np) para operações matemáticas em arrays multidimensionais e pandas (pd) para trabalhar com dados em diversos formatos, como CSV. A biblioteca tqdm é utilizada para exibir o progresso do processamento do vídeo.

O vídeo a ser processado é carregado utilizando a função cv2.VideoCapture(), que recebe como parâmetro o caminho do vídeo. Em seguida, as imagens de treinamento do Lula e do Bolsonaro são carregadas utilizando a função face_recognition.load_image_file(). Os rostos nas imagens de treinamento são codificados utilizando face_recognition.face_encodings() e os encodings são armazenados em uma lista.

A função detecta_faces() é responsável por detectar as faces presentes em um determinado quadro do vídeo. Ela redimensiona o quadro, converte-o para o formato RGB, utiliza o algoritmo de detecção facial CNN para localizar as faces e em seguida codifica as faces encontradas utilizando face_recognition.face_encodings(). Os nomes das faces são determinados comparando os encodings das faces encontradas com os encodings das imagens de treinamento utilizando face_recognition.compare_faces(). Em seguida, os retângulos e os nomes das faces são desenhados no quadro.

A função processa_frames() é responsável por ler cada quadro do vídeo, chamar a função detecta_faces() para detectar as faces presentes, desenhar os retângulos e os nomes das faces, e exibir o resultado em uma janela de vídeo. O tempo de cada pessoa reconhecida (Lula e Bolsonaro) e o tempo total do vídeo são calculados e armazenados em variáveis. Após a conclusão do processamento do vídeo, um DataFrame é criado com as informações de tempo e é exibido na saída.

No final do código, a função processa_frames() é chamada para iniciar o processamento do vídeo.

Certifique-se de que as bibliotecas necessárias estejam instaladas, siga as etapas de uso corretamente e utilize as imagens de treinamento apropriadas para obter resultados precisos no reconhecimento facial.





