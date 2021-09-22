import cv2

image_path = 'data//sudoku_images//su0.png'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()

