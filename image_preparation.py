import cv2
import numpy as np


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners


def crop_image(img):
    corners = get_corners(img)
    x = np.array(corners)[:, 0]
    y = np.array(corners)[:, 1]
    crop_img = img[y.min():y.max(), x.min():x.max()]
    return crop_img


def process(img_path, dim=600):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)

    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)

    crop_img = crop_image(dilated)
    resized_img = cv2.resize(crop_img, (dim, dim))
    cv2.imshow('resized_img', resized_img)

    # TODO: remove salt and pepper noise
    median_blur = cv2.medianBlur(resized_img, 3)
    # gauss = cv2.GaussianBlur(median_blur, (3, 3), 0)
    cv2.imshow('median_blur', median_blur)

    cv2.waitKey()

    return resized_img


if __name__ == "__main__":
    all_paths = ['su0.png', 'su1.png', 'su2.jpg']
    image_path = f'data//sudoku_images//{all_paths[2]}'
    processed_image = process(image_path)

    cv2.imshow('processed_image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# TODO: finish the clean pipeline
