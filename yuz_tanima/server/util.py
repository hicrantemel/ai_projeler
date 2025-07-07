import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
from tensorflow.keras.models import load_model

__class_name_to_number = {}
__class_number_to_name = {}

__model = None



def classify_image(image_base64_data, file_path=None):

    # Görüntüyü kırpıyoruz (2 göz var mı kontrolü ile)
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        # Görüntüyü 64x64 boyutuna yeniden boyutlandır
        scalled_raw_img = cv2.resize(img, (64, 64))

        # Gri tonlamaya çeviriyoruz
        gray_img = cv2.cvtColor(scalled_raw_img, cv2.COLOR_BGR2GRAY)

        # Modelin beklediği formata getiriyoruz: (1, 64, 64, 1)
        final_img = gray_img.reshape(1, 64, 64, 1).astype(float) / 255.0

        # Modelden tahmin al
        prediction = __model.predict(final_img)
        class_probabilities = prediction[0]
        
        # Sonuçları formatlayalım
        result.append({
            'class': class_number_to_name(np.argmax(class_probabilities)),
            'class_probability': np.around(class_probabilities * 100, 2).tolist(),
            'class_dictionary': __class_name_to_number
        })

    return result


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        __model = load_model('./artifacts/face_detect_model.h5')
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
   
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            cropped_faces.append(roi_color)
    return cropped_faces



if __name__ == '__main__':
    load_saved_artifacts()

