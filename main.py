import cv2
import numpy as np

image_path = 'data//sudoku_images//su0.png'


# -----------------------
# 1. Pre Processing the Image
# -----------------------
def preprocessing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow("img", img)

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


processed_image = preprocessing(image_path)
print(processed_image.shape)


# -----------------------
# 2. Sudoku Extraction
# -----------------------

def get_grid_coordinates(img, show=False):
    grid_img = 255 * np.ones((img.shape[0], img.shape[1], 3), np.uint8)  # white frame
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f'len contours: {len(contours)}')
    grid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(grid_img, contour, -1, (0, 0, 0), 1)

            # =------------------------
            coordinates = contour.ravel()
            # print('coordinates: ', len(coordinates))
            x_coord = coordinates[::2]
            # print('x_coord: ', len(x_coord))
            y_coord = coordinates[1:][::2]
            # print('y_coord: ', len(y_coord))

            x_min = x_coord.min()
            x_max = x_coord.max()

            y_min = y_coord.min()
            y_max = y_coord.max()

            # =------------------------

            grid_contours.append([x_min, x_max, y_min, y_max])
    if show:
        cv2.imshow('Contours', grid_img)
        cv2.waitKey(0)
    return grid_contours


grid_contours = get_grid_coordinates(processed_image, show=False)
print('grid_contours: ', grid_contours)
print('end')


# draw four corners:
def plot_dots_on_grid_corners(img, dot_size=5):
    grid_cntrs = get_grid_coordinates(img, show=False)
    for i in range(len(grid_cntrs)):
        cv2.circle(img, (grid_cntrs[i][0], grid_cntrs[i][2]), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (grid_cntrs[i][1], grid_cntrs[i][2]), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (grid_cntrs[i][0], grid_cntrs[i][3]), dot_size, [0, 0, 255], -1)
        cv2.circle(img, (grid_cntrs[i][1], grid_cntrs[i][3]), dot_size, [0, 0, 255], -1)
    cv2.imshow("dots on grid corners", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


plot_dots_on_grid_corners(processed_image)

# todo: reuse coordinates for indicating wheather box is empty or with number

# def get_grid_lines(img, horizontal_size = 11, vertical_size = 11):
#     # Create structuring elements
#     horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
#     verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
#
#     # Morphological opening
#     horizontal_mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontalStructure)
#     vertical_mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, verticalStructure)
#
#     # Outputs
#     cv2.imshow('img', img)
#     cv2.imshow('horizontal_mask', horizontal_mask)
#     cv2.imshow('vertical_mask', vertical_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return horizontal_mask, vertical_mask
#
# horizontal_mask, vertical_mask = get_grid_lines(processed_image, horizontal_size = 30, vertical_size = 30)


# TODO: make an image clean at first: find four corners and crop
def get_grid_boxes(img, show=False):
    """Goes row by row and stores individual sudoku grid boxes in the list."""
    box_height = int(img.shape[0] / 9)  # one small rect_num
    box_width = int(img.shape[1] / 9)
    print(f'box_height: {box_height}, box_width: {box_width}')
    x, y = 0, 0  # the start point
    grid_boxes = []
    for rect_num in range(9 * 9):  # 9x9 sudoku game
        print(f'[y:y+box_height, x:x + box_width]: [{y}:{y + box_height}, {x}:{x + box_width}]')
        crop_img = processed_image[y:y + box_height, x:x + box_width]
        grid_boxes.append(crop_img)
        print(crop_img)
        x = x + box_width
        if (rect_num + 1) % 9 == 0:
            print('swap')
            y = y + box_height
            x = 0
        if show:
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)
    return grid_boxes


def clean_grid_edges(grid_boxes):
    for box in grid_boxes:
        print(box)


grid_boxes = get_grid_boxes(processed_image, show=True)

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
