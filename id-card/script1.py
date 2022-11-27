import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

def detect_face(image,data_dir = str(DATA_DIR),  x_eps=0.3, y_eps=0.4):
    """
        Detect and return (x,y,w,h) of the wider face on an image
        x,y - top left position of detected face rectangle
        w,h - width and height
        Raise an IndexError when no face detected

        Before use, make sure to provide openCv trained models directory (defaut is ./data/)
    """
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    face_cascade_model_file = os.path.join(data_dir,'haarcascade_frontalface_alt.xml')
    face_cascade = cv.CascadeClassifier(face_cascade_model_file)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )
    if len(faces) != 0:
        if len(faces) > 1:
            faces = sorted(faces, key = lambda face : face[2] * face[3], reverse=True)
        
        # image_width = image.shape[1]
        # image_height = image.shape[0]
        
        x,y,w,h = faces[0]
        x -= int(x_eps*w)
        y -= int(y_eps*h)
        w = int(w*(1+2*x_eps))
        h = int(h*(1+2*y_eps))
        return (x,y,w,h)

    else:
        raise IndexError('No relevant face detected')

def replace_face(source,target):
    """
        Detect and replace unique relevant face in source by target image
        return source with replaced face
    """
    source_copy = source.copy()
    x,y,w,h = detect_face(source_copy,x_eps=0.25, y_eps=0.4)
    resized_target = cv.resize(target,(w,h),interpolation = cv.INTER_AREA)
    source_copy[y:y+h,x:x+w] = resized_target[0:h,0:w]
    return source_copy


if __name__ == '__main__':
            
    inputs_dir = BASE_DIR / 'inputs'
    output_dir = BASE_DIR / 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    cards_dir = inputs_dir / 'cards'
    target_dir = inputs_dir / 'targets'

    for s_file in cards_dir.iterdir():
        source = cv.imread(str(s_file))
        i = 1
        for t_file in target_dir.iterdir():
            target = cv.imread(str(t_file))

            try:
                replaced = replace_face(source,target)
                cv.imwrite(os.path.join(output_dir,f'{i}replaced_{s_file.name}'),replaced)
                i += 1
                print(f'Replacement on {s_file.name} succeded!')
            except:
                print(f'Replacement on {s_file.name} failed!')