# Image postprocessing (not mine functions) need revision
# https://github.com/AliShazly/sudoku-py/blob/00cc82dd3add666a626b3320f089b90e5271aca1/main.py#L276
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def subdivide(img, divisions=9):
    height, _ = img.shape[:2]
    box = height // divisions
    if len(img.shape) > 2:
        subdivided = img.reshape(height // box, box, -1, box, 3).swapaxes(1, 2).reshape(-1, box, box, 3)
    else:
        subdivided = img.reshape(height // box, box, -1, box).swapaxes(1, 2).reshape(-1, box, box)
    return [i for i in subdivided]


def stitch_img(img_arr, img_dims):
    result = Image.new('RGB' if len(img_arr[0].shape) > 2 else 'L', img_dims)
    box = [0, 0]
    for img in img_arr:
        pil_img = Image.fromarray(img)
        result.paste(pil_img, tuple(box))
        if box[0] + img.shape[1] >= result.size[1]:
            box[0] = 0
            box[1] += img.shape[0]
        else:
            box[0] += img.shape[1]
    return np.array(result)


def transform(pts, img):  # TODO: Spline transform, remove this
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped


def put_solution(img_arr, soln_arr, unsolved_arr):
    solutions = np.array(soln_arr).reshape(81)
    unsolveds = np.array(unsolved_arr).reshape(81)
    paired = list((zip(solutions, unsolveds, img_arr)))
    img_solved = []
    for solution, unsolved, img in paired:
        if solution == unsolved:
            img_solved.append(img)
            continue
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        fnt = ImageFont.truetype('fonts\\FreeMono.ttf', img_h)
        font_w, font_h = draw.textsize(str(solution), font=fnt)
        draw.text(((img_w - font_w) / 2, (img_h - font_h) / 2 - img_h // 10), str(solution),
                  fill=((0, 0, 165) if len(img.shape) > 2 else 0), font=fnt)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_solved.append(cv2_img)
    return img_solved


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0],
                           [img.shape[1] - 1, img.shape[0] - 1],
                           [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img
