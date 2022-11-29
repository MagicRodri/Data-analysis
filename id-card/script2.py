#! usr/env

from pathlib import Path
from retinaface import RetinaFace
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
import math
from PIL import Image


def detect_face(image ,x_eps=0.35, y_eps=0.35):


    resp = RetinaFace.detect_faces(image)
    
    facial_area = resp.get('face_1').get('facial_area')
    x1,y1,x2,y2 = facial_area
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    x1 -= int(x_eps*w)
    y1 -= int(y_eps*h)
    w = int(w*(1+2*x_eps))
    h = int(h*(1+2*y_eps))
    return (x1,y1,w,h)

def draw_dectected_face(image):
    """
    """
    x,y,w,h = detect_face(image)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(image)

def align_face(image):
  """
  """
  resp = RetinaFace.detect_faces(image)
  face = resp['face_1']
  x1 , y1 = face['landmarks']['right_eye']
  x2 , y2 = face['landmarks']['left_eye']
  a = abs(y1 - y2)
  b = abs(x2 - x1)
  c = math.sqrt(a**2 + b**2)
  cos_alpha = (b**2 + c**2 - a**2) / (2*b*c)
  alpha = np.arccos(cos_alpha)
  alpha_degrees = np.degrees(alpha)
  pillow_img = Image.fromarray(image)
  aligned_img = np.array(pillow_img.rotate(alpha_degrees))

  return aligned_img

def replace_face(source,target,*, align=True , clone=True):
    """
        Detect and replace unique relevant face in source by target image
        return source with replaced face
    """
    if align:
      source = align_face(source)
      target = align_face(target)

    x,y,w,h = detect_face(source)
    resized_target = cv.resize(target,(w,h),interpolation = cv.INTER_AREA)

    if clone:
      resized_target_mask = 250 * np.ones(resized_target.shape, resized_target.dtype) # white mask
      center = (x+w//2,y+h//2)
      # Clone seamlessly.
      output = cv.seamlessClone(resized_target, source, resized_target_mask, center, cv.NORMAL_CLONE)
      return output

    print(resized_target.shape[0],h)
    # assert(resized_target.shape[0] == h and resized_target.shape[0] == w )
    source[y:y+h,x:x+w] = resized_target[:,:]
    return source