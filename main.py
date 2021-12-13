import logging
import os
import time
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.optimizers import Adam

import ground_truth as gt
from digit_recognition import define_model
from digit_recognition import prepare_data
from image_postprocessing import inverse_perspective, put_solution, transform, stitch_img, subdivide
from image_preparation import preprocess, get_grid_mask
from sudoku_solver import sudoku, print_sudoku


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
        if area > 1000:
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

    if len(grid_contours_coord) == 81:  # all sudoku 9*9 grids were found
        return grid_contours_coord
    else:
        return False


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

    # print_multi_bounding_boxes(grid_boxes_img)  # todo: remove
    if len(grid_boxes_img) == 81:
        return grid_boxes_img
    else:
        return False


def print_multi_bounding_boxes(grid_boxes_img):
    # horizontally:
    horizontal_stack = []
    for ind in range(len(grid_boxes_img)):
        if ind == 0:
            imstack = grid_boxes_img[ind]
        elif ind % 9 == 0:
            imstack = grid_boxes_img[ind]
        else:
            imstack = np.hstack((imstack, grid_boxes_img[ind]))
            if imstack.shape[1] == 252:
                horizontal_stack.append(imstack)

    # vertically:
    for ind in range(len(horizontal_stack)):
        if ind == 0:
            imstack = horizontal_stack[ind]
        else:
            imstack = np.vstack((imstack, horizontal_stack[ind]))

    cv2.imshow('stack', imstack)
    cv2.waitKey(0)


def invert_colors(grid_boxes_img):
    """Inverts the colors and leaves a black background and white numbers to match the training data."""
    inverted_boxes = []
    for img in grid_boxes_img:
        inverted_img = cv2.bitwise_not(img)
        inverted_boxes.append(inverted_img)
    return inverted_boxes


def predict_digits(grid_boxes_img, model_name, num_classes=10, learning_rate=0.001):
    """Predicts each digit of the cell by using the CNN model."""
    grid_boxes_img_prep = prepare_data(np.array(grid_boxes_img))

    # load trained model
    model = define_model(model_name, num_classes)
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(f"outputs//digit_recognition//{model_name}//saved_model.h5")

    # use model to predict digits:
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


def validate_sudoku_digits(digit_matrix):
    """Sudoku rows and columns should not contain duplicate numbers."""
    is_correct = True
    for row in digit_matrix:
        (unique, counts) = np.unique(row, return_counts=True)
        for value, count in zip(unique, counts):
            if value != 0:
                if count != 1:
                    print(f'row validation: value {value} repeats {count} times!')
                    is_correct = False
    for column in digit_matrix.T:
        (unique, counts) = np.unique(column, return_counts=True)
        for value, count in zip(unique, counts):
            if value != 0:
                if count != 1:
                    print(f'columns validation: value {value} repeats {count} times!')
                    is_correct = False
    return is_correct


def draw_results_on_image(img_path, digit_matrix, indexes_with_empty_values, show=True):
    image = Image.open(img_path)

    d = ImageDraw.Draw(image)
    font = ImageFont.truetype('fonts\\FreeMono.ttf', 35)
    coord_x = 12
    coord_y = 5
    step = 40
    for row_ind, row_values in enumerate(digit_matrix):
        for col_ind, value in enumerate(row_values):
            if ([row_ind, col_ind] == indexes_with_empty_values).all(1).any():
                fill = (0, 0, 165)
                d.text((coord_x, coord_y), str(value), fill=fill, font=font)
            else:  # todo: do not show in the end!
                fill = 0
                d.text((coord_x, coord_y), str(value), fill=fill, font=font)
            coord_x += step
        coord_x = 12
        coord_y += step
    # Converting from PIL to cv2 for output
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
    return image


def detect_video(video_path, video_output_path, model_name, video_output_format='MP4V'):
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 600
    height = 600
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*video_output_format)
    out = cv2.VideoWriter(video_output_path, codec, fps, (width, height))

    while True:
        _, frame = vid.read()
        start_time = time.time()
        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

        warped_inverse = main(image_path=None, img=frame, model_name=model_name, gt_array=None, from_path=False,
                              show=False)
        print(f'frame time: {round((time.time() - start_time) / 60, 2)} min.')

        if video_output_path:
            out.write(warped_inverse)
            logging.warning('video frame saved!')

        cv2.imshow('Recording...', warped_inverse)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


def main(image_path, img, model_name, gt_array, from_path=True, show=True):
    # --------------------------------------------
    processed_image, corners, img = preprocess(image_path, img=img, dim=600, from_path=from_path)
    grid_mask = get_grid_mask(processed_image)

    grid_contours = get_grid_coordinates(grid_mask, show=False)
    if grid_contours:
        grid_boxes = get_grid_boxes(processed_image, grid_contours, dim=28, show=False)
        # --------------------------------------------
        digit_matrix = predict_digits(grid_boxes, model_name, num_classes=10, learning_rate=0.001)
        original_matrix = digit_matrix.copy()
        indexes_with_empty_values = np.argwhere(np.isin(digit_matrix, [0]))
        print('digit_matrix: ')
        print(digit_matrix)
        is_correct = validate_sudoku_digits(digit_matrix)
        if is_correct:
            if gt_array:
                check_sudoku_accuracy(digit_matrix, gt_array)

            # --------------------------------------------
            is_solution = sudoku(digit_matrix, 0, 0)
            if is_solution:
                print_sudoku(digit_matrix)
            else:
                print('Solution does not exist')
            # --------------------------------------------
            # postprocessing
            warped_img = transform(corners, img)
            subd = subdivide(warped_img)
            subd_soln = put_solution(subd, digit_matrix, original_matrix)
            warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
            warped_inverse = inverse_perspective(warped_soln, img.copy(), np.array(corners))

            if show:
                cv2.imshow('warped_inverse', warped_inverse)
                cv2.waitKey(0)
            # --------------------------------------------
            # image = draw_results_on_image(image_path, digit_matrix, indexes_with_empty_values, show=show)
            return warped_inverse
        else:
            return img
    else:
        return img


if __name__ == "__main__":
    # 1. detect on image:
    all_paths = ['su0.png', 'su1.png', 'su2.jpg', 'su4.jpg', 'su5.jpg', 'su6.jpg', 'su7.jpg']
    gt_files = [gt.su0, gt.su1, gt.su2, gt.su4, gt.su5, gt.su6]
    all_paths = os.listdir('data/sudoku_images/')
    for i in range(len(all_paths)):
        if all_paths[i].endswith(".jpg"):
            image_path = f'data/sudoku_images/{all_paths[i]}'
            start_time = time.time()
            warped_inverse = main(image_path, img=None, model_name='cnn_architecture_7', gt_array=None, from_path=True,
                                  show=False)
            filename = os.path.join(os.path.split(image_path)[0], 'saved_images', os.path.split(image_path)[1])
            cv2.imwrite(filename, warped_inverse)
            print(f"--- {i} - Execution time: {round((time.time() - start_time), 2)} sec. ---")

    # # 2. detect on video:
    # start_time = time.time()
    # video_path = 'data/sudoku_images/video/crop_video3.mp4'
    # video_output_path = 'data/sudoku_images/video/crop_video3_saved.avi'
    # detect_video(video_path, video_output_path, model_name='cnn_architecture_7', video_output_format='MJPG')
    # print(f"--- Execution time: {round((time.time() - start_time), 2)} sec. ---")
