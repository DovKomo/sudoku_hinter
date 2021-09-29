# Some functions are taken from:
# https://github.com/AliShazly/sudoku-py/blob/00cc82dd3add666a626b3320f089b90e5271aca1/main.py#L276

import cv2
import numpy as np
from skimage import morphology


def get_largest_corners(img):
    """Finds the largest contour and returns 4 corners."""
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest = np.squeeze(contours[0])

    sums = [sum(i) for i in largest]
    differences = [i[0] - i[1] for i in largest]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest[top_left], largest[top_right], largest[bottom_left], largest[bottom_right]]
    return corners


def crop_image(img, corners):
    """Crops the image to predefined angles."""
    x = np.array(corners)[:, 0]
    y = np.array(corners)[:, 1]
    crop_img = img[y.min():y.max(), x.min():x.max()]
    return crop_img


def transform(img, corners):
    """Applies the perspective transformation to an image."""
    corners = np.float32(corners)
    top_l, top_r, bot_l, bot_r = corners[0], corners[1], corners[2], corners[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    transform_matrix = cv2.getPerspectiveTransform(corners, dim)
    warped = cv2.warpPerspective(img, transform_matrix, (square, square))
    return warped


def remove_small_dots(dilated_img, min_size=2, connectivity=2):
    """Removes small white dots."""
    remove_obj = morphology.remove_small_objects(dilated_img.astype(bool), min_size=min_size,
                                                 connectivity=connectivity).astype(int)
    mask_x, mask_y = np.where(remove_obj == 0)  # black out pixels
    dilated_img[mask_x, mask_y] = 0
    return dilated_img


def get_grid_lines(img, length=12, horizontal=True):
    """Gets an image of vertical or horizontal grid lines."""
    index = 0
    if horizontal:
        index = 1

    line = np.copy(img)
    cols = line.shape[index]
    line_size = cols // length

    struc_size = (line_size, 1)
    if horizontal:
        struc_size = (1, line_size)

    line_structure = cv2.getStructuringElement(cv2.MORPH_RECT, struc_size)
    line = cv2.erode(line, line_structure)
    line = cv2.dilate(line, line_structure)
    return line


def create_grid_mask(vertical, horizontal):
    """Creates a grid mask of vertical and horizontal lines."""
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(im, pts):
        im = np.copy(im)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(im, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return im

    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask


def get_grid_mask(img):
    """Returns the image of the grid mask."""
    vertical_lines = get_grid_lines(img, length=12, horizontal=False)
    horizontal_lines = get_grid_lines(img, length=12, horizontal=True)
    mask = create_grid_mask(vertical_lines, horizontal_lines)
    return mask


def preprocess(img_path, dim=600):
    """Performs all image preprocessing steps."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (dim, dim))
    # -------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, thresh)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    # -------------------------------------
    corners = get_largest_corners(dilated)
    warped = transform(dilated, corners)
    # -------------------------------------
    new_corners = get_largest_corners(warped)
    crop_img = crop_image(warped, new_corners)
    resized_img = cv2.resize(crop_img, (dim, dim))
    # -------------------------------------
    # remove salt and pepper noise
    removed_dots = remove_small_dots(resized_img, min_size=200, connectivity=10)
    median_blur = cv2.medianBlur(removed_dots, 3)
    return median_blur


if __name__ == "__main__":
    all_paths = ['su0.png', 'su1.png', 'su2.jpg']
    image_path = f'data//sudoku_images//{all_paths[2]}'
    processed_image = preprocess(image_path, dim=600)
    grid_mask = get_grid_mask(processed_image)
    cv2.imshow('processed_image', processed_image)
    cv2.imshow('grid_mask', grid_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
