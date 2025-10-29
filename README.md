# --- 1. OpenCV 설치 (Colab용) ---
!pip install opencv-python-headless

# --- 2. 이미지 업로드 ---
from google.colab import files
uploaded = files.upload()  # 파일 선택

import cv2, numpy as np
from matplotlib import pyplot as plt

# --- 3. 업로드한 이미지 읽기 ---
filename = list(uploaded.keys())[0]
img = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 4. 얼굴 인식 ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# --- 5. 얼굴 위치 표시 ---
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# --- 6. 결과 출력 ---
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
