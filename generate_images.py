import cv2
import os
import numpy as np

DEFAULT_HEIGHT = 28
DEFAULT_WIDTH = 28


def create_blank_image(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    blank_image = np.zeros((height, width), np.uint8)
    return blank_image


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def create_minus_image():
    # Create blank image
    blank_image = create_blank_image()

    rect_width = int(DEFAULT_WIDTH/2)
    rect_height = int(DEFAULT_HEIGHT/5)
    final_rect_height = int(rect_height + 4 * rect_height)
    final_rect_width = int(rect_width + 3)

    blank_image[rect_height:final_rect_height, rect_width: final_rect_width] = 255 * \
        np.ones((final_rect_height - rect_height, final_rect_width - rect_width))

    minus = np.rot90(blank_image)
    return minus



minus = create_minus_image()
show_image(minus)