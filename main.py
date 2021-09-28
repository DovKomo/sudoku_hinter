import cv2
import numpy as np
from tensorflow import keras

import ground_truth as gt
from digit_recognition import prepare_data


# ----------------------------
# 1. Pre Processing the Image
# ----------------------------
def preprocessing(path, show=False):
    """Performs image preprocessing steps."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (361, 361))  # TODO: cropping ...

    # # gaussian blur:
    # img = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Load image, grayscale, and adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # invert colors:
    process1 = cv2.bitwise_not(thresh, thresh)

    # erotion:
    kernel = np.array([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], np.uint8)
    # process = cv2.dilate(process, kernel)
    process = cv2.erode(process1, kernel)

    if show:
        cv2.imshow("original image", img)
        # cv2.imshow("thresholded image", thresh)
        # cv2.imshow("bitwise image", process1)
        cv2.imshow("processed image", process)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return process


# -----------------------
# 2. Sudoku Extraction
# -----------------------

def get_grid_coordinates(img, show=False):
    """Stores grid coordinates in the list if the area > 1000, so that each cell is isolated."""
    grid_img = 255 * np.ones((img.shape[0], img.shape[1], 3), np.uint8)  # white frame
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    grid_contours_coord = []

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

            grid_contours_coord.append([x_min, x_max, y_min, y_max])

    if show:
        cv2.imshow('Contours', grid_img)
        cv2.waitKey(0)

    assert len(grid_contours_coord) == 81  # all sudoku 9*9 grids were found

    return grid_contours_coord


def plot_dots_on_grid_corners(img, dot_size=5):
    """Draw the four corners of the grid boxes to make sure the coordinates are correct."""
    grid_contours_coord = get_grid_coordinates(img, show=False)
    for cnt in grid_contours_coord:
        x_min, x_max, y_min, y_max = cnt
        cv2.circle(img, (x_min, y_min), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_max, y_min), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_min, y_max), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (x_max, y_max), dot_size, [0, 0, 255], -1)
    cv2.imshow("Dots on grid corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_grid_boxes(img, grid_contours_coord, dim=28, show=False):
    """Goes row by row from the bottom right to the left and up, and stores individual sudoku grid boxes in the list."""
    grid_boxes_img = []
    for cnt in grid_contours_coord:
        x_min, x_max, y_min, y_max = cnt
        crop_img = img[y_min:y_max, x_min:x_max]
        # resize image
        resized_crop_img = cv2.resize(crop_img, (dim, dim), interpolation=cv2.INTER_AREA)
        grid_boxes_img.append(resized_crop_img)
        if show:
            cv2.imshow("cropped", resized_crop_img)
            cv2.waitKey(0)
    return grid_boxes_img


def invert_colors(grid_boxes_img):
    """Inverts the colors and leaves a black background and white numbers to match the training data."""
    inverted_boxes = []
    for img in grid_boxes_img:
        inverted_img = cv2.bitwise_not(img)
        inverted_boxes.append(inverted_img)
    return inverted_boxes


def predict_digits(grid_boxes_img, saved_model='outputs//digit_recognition//saved_model.h5'):
    """Predicts each digit of the cell by using the CNN model."""
    # before using the model - important to invert colors
    grid_boxes_img_inv = invert_colors(grid_boxes_img)
    grid_boxes_img_prep = prepare_data(np.array(grid_boxes_img_inv))

    model = keras.models.load_model(saved_model)
    predicted_numbers = model.predict(grid_boxes_img_prep)
    predicted_numbers = np.argmax(predicted_numbers, axis=1)  # highest probability

    digits = []
    for img, i in zip(grid_boxes_img_prep, range(len(grid_boxes_img))):
        if np.max(img) != 1:  # all black (no number when inverted image)
            number = 0
        else:
            number = predicted_numbers[i]
            if number == 0:  # 0 can not be predicted
                print('miss-predicted as 0: ', number)
        digits.append(number)

    digits_reversed = digits[::-1]
    digits_matrix = np.reshape(digits_reversed, (9, 9))

    return digits_matrix


def check_sudoku_accuracy(predicted_digits, true_digits):
    """Calculates accuracy of one sudoku image."""
    same_digits = np.count_nonzero(predicted_digits == true_digits)
    all_digits = len(np.ravel(predicted_digits))
    acc = same_digits / all_digits
    print(f'sudoku accuracy: {round(acc * 100, 2)}, miss-predicted: {all_digits - same_digits}')
    return acc


if __name__ == "__main__":
    all_paths = ['su0.png', 'su1.png', 'su2.jpg']
    image_path = f'data//sudoku_images//{all_paths[2]}'
    processed_image = preprocessing(image_path, show=True)
    print(f'image shape: {processed_image.shape}')
    grid_contours = get_grid_coordinates(processed_image, show=True)
    # plot_dots_on_grid_corners(processed_image)
    grid_boxes = get_grid_boxes(processed_image, grid_contours, dim=28, show=False)

    # --------------------------------------------
    digit_matrix = predict_digits(grid_boxes)
    print('digit_matrix: ', digit_matrix)
    check_sudoku_accuracy(digit_matrix, gt.su2)

# solve sudoku:
# https://www.askpython.com/python/examples/sudoku-solver-in-python
# https://towardsdatascience.com/solve-sudoku-using-linear-programming-python-pulp-b41b29f479f3
# https://liorsinai.github.io/coding/2020/07/27/sudoku-solver.html

# more:
# https://towardsdatascience.com/solve-sudokus-automatically-4032b2203b64
# https://github.com/prajwalkr/SnapSudoku
