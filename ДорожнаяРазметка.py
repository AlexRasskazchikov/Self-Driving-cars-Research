import cv2
import numpy as np
import sys

size = (720, 720)

if __name__ == '__main__':
    def nothing(*arg):
        pass

cap = cv2.VideoCapture("test_footage.mp4")

cv2.namedWindow("result")  # создаем главное окно  # создаем окно настроек
cv2.namedWindow("setting2") # создаем окно настроек 2
cv2.namedWindow("setting3")


cv2.createTrackbar('h1', 'setting2', 0, 255, nothing)
cv2.createTrackbar('s1', 'setting2', 0, 255, nothing)
cv2.createTrackbar('v1', 'setting2', 155, 255, nothing)
cv2.createTrackbar('h2', 'setting2', 199, 255, nothing)
cv2.createTrackbar('s2', 'setting2', 49, 255, nothing)
cv2.createTrackbar('v2', 'setting2', 255, 255, nothing)
cv2.createTrackbar('H_Up', 'setting3', 318, 720, nothing)
cv2.createTrackbar('W_Up', 'setting3', 318, 720, nothing)
cv2.createTrackbar('H_Down', 'setting3', 720, 720, nothing)
cv2.createTrackbar('W_Down', 'setting3', 720, 720, nothing)


crange = [0, 0, 0, 0, 0, 0]

t_height = 350

dst = np.float32([[0, size[0]], [size[1], size[0]], [size[1], 0], [0, 0]])

if not cap.isOpened():
    raise FileNotFoundError("Путь указан неверно.")

while cv2.waitKey(1) != 27:
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'setting2')
    s1 = cv2.getTrackbarPos('s1', 'setting2')
    v1 = cv2.getTrackbarPos('v1', 'setting2')
    h2 = cv2.getTrackbarPos('h2', 'setting2')
    s2 = cv2.getTrackbarPos('s2', 'setting2')
    v2 = cv2.getTrackbarPos('v2', 'setting2')
    poly_height = cv2.getTrackbarPos('H_Up', 'setting3')
    poly_down = cv2.getTrackbarPos('H_Down', 'setting3')
    poly_up_width = cv2.getTrackbarPos('W_Up', 'setting3')
    poly_down_width = cv2.getTrackbarPos('W_Down', 'setting3')

    polygon = np.float32([[size[0] // 2 - poly_down_width, 720 - poly_down],
                          [size[0] // 2 + poly_down_width, 720 - poly_down],
                          [size[0] // 2 + poly_up_width // 2, size[1] - poly_height],
                          [size[0] // 2 - poly_up_width // 2, size[1] - poly_height]])
    poly_draw = np.array(polygon, dtype=np.int32)

    # формируем начальный и конечный цвет фильтра
    h_min, h_max = np.array((h1, s1, v1), np.uint8), np.array((h2, s2, v2), np.uint8)

    # Читаем камеру.
    ret, frame = cap.read()

    # Если проблемы с подключением.
    if not ret:
        raise TimeoutError("Конец Видео.")

    # Меняем размер картинки.
    resized = cv2.resize(frame, (size[1], size[0]))
    cv2.polylines(resized, [poly_draw], True, (255, 0, 0))

    # Динамичное редактирование цветогого спектра.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    allBinary = cv2.inRange(hsv, h_min, h_max)


    M = cv2.getPerspectiveTransform(polygon, dst)
    warped = cv2.warpPerspective(allBinary, M, (size[1], size[0]), flags=cv2.INTER_LINEAR)
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    midpoint = histogram.shape[0] // 2
    IndWhitestCoulumnL = np.argmax(histogram[:midpoint])
    IndWhitestCoulumnR = np.argmax(histogram[midpoint:]) + midpoint
    warped_visual = warped.copy()
    cv2.line(warped_visual, (IndWhitestCoulumnL, 0), (IndWhitestCoulumnL, warped_visual.shape[0]), 110, 2)
    cv2.line(warped_visual, (IndWhitestCoulumnR, 0), (IndWhitestCoulumnR, warped_visual.shape[0]), 110, 2)

    nwindows, window_half = 20, 30
    win_height = np.int(warped.shape[0] / nwindows)

    XCenterLeftWindow = IndWhitestCoulumnL
    XCenterRightWindow = IndWhitestCoulumnR

    left_lane_inds = np.array([], dtype=np.int16)
    right_lane_inds = np.array([], dtype=np.int16)

    out_img = np.dstack((warped, warped, warped))

    nonzero = warped.nonzero()
    WhitePixelY = np.array(nonzero[0])
    WhitePixelX = np.array(nonzero[1])

    for window in range(nwindows):
        win_y1 = warped.shape[0] - (window + 1) * win_height
        win_y2 = warped.shape[0] - window * win_height

        left_win_x1 = XCenterLeftWindow - window_half
        left_win_x2 = XCenterLeftWindow + window_half
        right_win_x1 = XCenterRightWindow - window_half
        right_win_x2 = XCenterRightWindow + window_half

        cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
        cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 50 + window * 21, 0), 2)

        good_left = ((win_y1 <= WhitePixelY) & (WhitePixelY <= win_y2) &
                     (left_win_x1 <= WhitePixelX) & (WhitePixelX <= left_win_x2)).nonzero()[0]

        good_right = ((WhitePixelY >= win_y1) & (WhitePixelY <= win_y2) & (WhitePixelX >= right_win_x1) &
                      (right_win_x2 >= WhitePixelX)).nonzero()[0]

        left_lane_inds = np.concatenate((left_lane_inds, good_left))
        right_lane_inds = np.concatenate((right_lane_inds, good_right))

        try:
            XCenterLeftWindow = np.int(np.mean(WhitePixelX[good_left]))
        except:
            pass

        try:
            XCenterRightWindow = np.int(np.mean(WhitePixelX[good_right]))
        except:
            pass

    out_img[WhitePixelY[left_lane_inds], WhitePixelX[left_lane_inds]] = [50, 50, 200]
    out_img[WhitePixelY[right_lane_inds], WhitePixelX[right_lane_inds]] = [50, 50, 200]

    """leftx = WhitePixelX[left_lane_inds]
    lefty = WhitePixelY[left_lane_inds]

    rigthx = WhitePixelX[right_lane_inds]
    righty = WhitePixelY[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rigthx, 2)

    center_fit = (left_fit + right_fit) // 2

    for ver_ind in out_img.shape[0]:
        gor_ind = (center_fit[0]) ^ (ver_ind ** 2) + center_fit[1] * ver_ind + center_fit[2]
        cv2.circle(out_img, (int(gor_ind), int(ver_ind)), 2, (255, 0, 255), 1)"""

    R = cv2.getPerspectiveTransform(dst, polygon)
    out_img = cv2.warpPerspective(out_img, R, (size[1], size[0]), flags=cv2.INTER_LINEAR)
    out_img = cv2.resize(out_img, (330, 330))
    resized = cv2.resize(resized, (330, 330))
    warped = cv2.resize(warped, (330, 330))
    cv2.imshow("result", out_img)
    cv2.imshow("standart", resized)
    cv2.imshow("warped", warped)

cv2.destroyAllWindows()
sys.exit()