import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Pre Processing the Image
# ----------------------------
def preprocessing(image_path):
    """Performs image preprocessing steps."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # cv2.imshow("img", img)

    # # gaussian blur:
    # img = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Load image, grayscale, and adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # invert colors:
    process = cv2.bitwise_not(thresh, thresh)

    # # dilation:
    kernel = np.array([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], np.uint8)
    # process = cv2.dilate(process, kernel)
    process = cv2.erode(process, kernel)

    # cv2.imshow("processed", process)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return process


# -----------------------
# 2. Sudoku Extraction
# -----------------------

def get_grid_coordinates(img, show=False):
    """Stores grid coordinates in the list if the area > 1000, so that each cell is isolated."""
    grid_img = 255 * np.ones((img.shape[0], img.shape[1], 3), np.uint8)  # white frame
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    grid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # print('area: ', area)
        if area > 1000:  # TODO: check if it's constant? 1368
            # print('include')
            cv2.drawContours(grid_img, contour, -1, (0, 0, 0), 1)

            coordinates = contour.ravel()

            x_coord = coordinates[::2]
            y_coord = coordinates[1:][::2]

            x_min = x_coord.min()
            x_max = x_coord.max()

            y_min = y_coord.min()
            y_max = y_coord.max()

            grid_contours.append([x_min, x_max, y_min, y_max])
    assert len(grid_contours) == 81  # all sudoku 9*9 grids were found

    if show:
        cv2.imshow('Contours', grid_img)
        cv2.waitKey(0)

    return grid_contours


def plot_dots_on_grid_corners(img, dot_size=5):
    """Draw the four corners of the grid boxes to make sure the coordinates are correct."""
    grid_contours = get_grid_coordinates(img, show=False)
    for cnt in grid_contours:
        x_min, x_max, y_min, y_max = cnt
        cv2.circle(img, (x_min, y_min), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_max, y_min), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_min, y_max), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_max, y_max), dot_size, [0, 0, 255], -1)
    cv2.imshow("Dots on grid corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_grid_boxes(img, grid_contours, show=False):
    """Goes row by row from the bottom right to the left and up, and stores individual sudoku grid boxes in the list."""
    grid_boxes_img = []
    for cnt in grid_contours:
        x_min, x_max, y_min, y_max = cnt
        crop_img = img[y_min:y_max, x_min:x_max]
        grid_boxes_img.append(crop_img)
        if show:
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)
    return grid_boxes_img


if __name__ == "__main__":
    image_path = 'data//sudoku_images//su0.png'
    processed_image = preprocessing(image_path)
    print(f'image shape: {processed_image.shape}')
    grid_contours = get_grid_coordinates(processed_image, show=False)
    # plot_dots_on_grid_corners(processed_image)
    grid_boxes_img = get_grid_boxes(processed_image, grid_contours, show=False)


# TODO: extract number of each of the grid box and get a matrix


# solve sudoku:
# https://www.askpython.com/python/examples/sudoku-solver-in-python
# https://towardsdatascience.com/solve-sudoku-using-linear-programming-python-pulp-b41b29f479f3
# https://liorsinai.github.io/coding/2020/07/27/sudoku-solver.html


# hand written digits:
#  https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

# more:
# https://towardsdatascience.com/solve-sudokus-automatically-4032b2203b64
# https://github.com/prajwalkr/SnapSudoku
