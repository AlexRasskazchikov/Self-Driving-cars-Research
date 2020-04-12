import cv2
image = "test_footage.mp4"

def nothing(*arg):
    pass

cap = cv2.VideoCapture(image)
size = (700, 700)
cv2.namedWindow("crop")  # создаем окно настроек 3

# Crop-слайдеры.
cv2.createTrackbar('Up', 'crop', 98, size[1] // 2, nothing)
cv2.createTrackbar('Down', 'crop', 56, size[1] // 2, nothing)
cv2.createTrackbar('Left', 'crop', 70, size[0] // 2, nothing)
cv2.createTrackbar('Right', 'crop', 40, size[0] // 2, nothing)

while cv2.waitKey(1) != 27:
    ret, frame = cap.read()

    #   Слайдеры, отвечающие за кроп.
    up_crop = cv2.getTrackbarPos('Up', 'crop')
    down_crop = size[1] - cv2.getTrackbarPos('Down', 'crop')
    left_crop = cv2.getTrackbarPos('Left', 'crop')
    right_crop = size[0] - cv2.getTrackbarPos('Right', 'crop')

    crop_img = frame[up_crop:down_crop, left_crop:right_crop]
    cv2.imshow("crop", crop_img)