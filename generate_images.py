import cv2
import os
import numpy as np
import scipy.ndimage

DEFAULT_HEIGHT = 100
DEFAULT_WIDTH = 100


def create_blank_image(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    blank_image = np.zeros((height, width), np.uint8)
    return blank_image


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_minus_image():
    # Create blank image
    blank_image = create_blank_image()

    start_height = int(DEFAULT_HEIGHT * 0.1)
    end_height = int(DEFAULT_HEIGHT * 0.9)
    rect_height = int(end_height - start_height)

    start_width = int(DEFAULT_WIDTH * 0.5 - 1.5)
    end_width = int(DEFAULT_WIDTH * 0.5 + 3)
    rect_width = int(end_width - start_width)

    blank_image[start_height:end_height, start_width: end_width] = 255 * \
        np.ones((rect_height, rect_width))

    minus = np.rot90(blank_image)

    minus = cv2.resize(minus, (28, 28))
    return minus


def create_plus_image():
    minus = create_minus_image()
    minus = np.rot90(minus)
    minus2 = create_minus_image()
    plus = np.bitwise_or(minus, minus2)
    return plus


def create_multiply_image():
    plus = create_plus_image()
    plus = scipy.ndimage.rotate(plus, 45.0)
    multiply = plus[8:32, 8:32]
    multiply = cv2.resize(multiply, (28, 28))    
    return multiply


minus = create_multiply_image()
show_image(minus)