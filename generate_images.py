import cv2
import numpy as np

def create_blank_image(height = 28, width = 28):
    blank_image = np.zeros((height, width, 1), np.uint8)
    return blank_image

