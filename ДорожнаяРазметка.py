import cv2
import numpy as np
import sys

size = (720, 720)

def nothing(*arg):
    pass

cam = 2
path = "test_footage.mp4"

frame_counter = 0

video_name = path

cap = cv2.VideoCapture(video_name)

cv2.namedWindow("result")  # создаем главное окно  # создаем окно настроек
cv2.namedWindow("setting2")  # создаем окно настроек 1
cv2.namedWindow("setting3")  # создаем окно настроек 2
cv2.namedWindow("crop1")  # создаем окно настроек 3

# Threshold слайдеры.
cv2.createTrackbar('h1', 'setting2', 0, 255, nothing)
cv2.createTrackbar('s1', 'setting2', 0, 255, nothing)
cv2.createTrackbar('v1', 'setting2', 155, 255, nothing)
cv2.createTrackbar('h2', 'setting2', 199, 255, nothing)
cv2.createTrackbar('s2', 'setting2', 49, 255, nothing)
cv2.createTrackbar('v2', 'setting2', 255, 255, nothing)

# Слайдеры, отвечающие за варп трапеции.
cv2.createTrackbar('1H', 'setting3', 291, 720, nothing)  # Высота верхнего основания.
cv2.createTrackbar('1W', 'setting3', 178, 720, nothing)  # Ширина верхнего основания.
cv2.createTrackbar('2H', 'setting3', 138, 720, nothing)  # Высота нижнего основания.
cv2.createTrackbar('2W', 'setting3', 577, 720, nothing)  # Ширина нижнего основания.
cv2.createTrackbar('WW', 'setting3', 30, 50, nothing)  # Ширина окошек слежения за полосой.
cv2.createTrackbar('WC', 'setting3', 20, 50, nothing)  # Количество окошек слежения за полосой.

# Crop-слайдеры.
cv2.createTrackbar('Up', 'crop1', 5, size[1] // 2, nothing)
cv2.createTrackbar('Down', 'crop1', 5, size[1] // 2, nothing)
cv2.createTrackbar('Left', 'crop1', 5, size[0] // 2, nothing)
cv2.createTrackbar('Right', 'crop1', 5, size[0] // 2, nothing)

crange = [0, 0, 0, 0, 0, 0]

# Координаты для варпа трапеции, углы картинки.
dst = np.float32([[0, size[0]], [size[1], size[0]], [size[1], 0], [0, 0]])

if not cap.isOpened():
    raise FileNotFoundError("Путь указан неверно.")

while cv2.waitKey(1) != 27:

    frame_counter += 1

    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap = cv2.VideoCapture(video_name)


    # считываем значения бегунков

    #   Threshold слайдеры.
    h1 = cv2.getTrackbarPos('h1', 'setting2')
    s1 = cv2.getTrackbarPos('s1', 'setting2')
    v1 = cv2.getTrackbarPos('v1', 'setting2')
    h2 = cv2.getTrackbarPos('h2', 'setting2')
    s2 = cv2.getTrackbarPos('s2', 'setting2')
    v2 = cv2.getTrackbarPos('v2', 'setting2')

    #   Слайдеры, отвечающие за варп трапеции.
    poly_height, poly_down = cv2.getTrackbarPos('1H', 'setting3'), cv2.getTrackbarPos('2H', 'setting3')
    poly_up_width, poly_down_width = cv2.getTrackbarPos('1W', 'setting3'), cv2.getTrackbarPos('2W', 'setting3')

    #   Слайдеры, отвечающие за настройку окошек слежения.
    WC, WW = cv2.getTrackbarPos('WC', 'setting3'), cv2.getTrackbarPos('WW', 'setting3')

    #   Слайдеры, отвечающие за кроп.
    up_crop = cv2.getTrackbarPos('Up', 'crop1')
    down_crop = size[1] - cv2.getTrackbarPos('Down', 'crop1')
    left_crop = cv2.getTrackbarPos('Left', 'crop1')
    right_crop = size[0] - cv2.getTrackbarPos('Right', 'crop1')

    # Выстраиваем трапецию по координатам, полученных с бегунков.
    polygon = np.float32([[size[0] // 2 - poly_down_width // 2, 720 - poly_down],
                          [size[0] // 2 + poly_down_width // 2, 720 - poly_down],
                          [size[0] // 2 + poly_up_width // 2, size[1] - poly_height],
                          [size[0] // 2 - poly_up_width // 2, size[1] - poly_height]])
    poly_draw = np.array(polygon, dtype=np.int32)

    # Формируем начальный и конечный цвет фильтра бинаризации изображения.
    h_min, h_max = np.array((h1, s1, v1), np.uint8), np.array((h2, s2, v2), np.uint8)

    # Читаем камеру.
    ret, frame = cap.read()

    # Если проблемы с подключением.
    if not ret:
        raise TimeoutError("Конец Видео.")

    # Меняем размер картинки.
    resized = cv2.resize(frame, (size[1], size[0]))
    cv2.polylines(resized, [poly_draw], True, (0, 0, 255), 3)

    # Динамичное редактирование цветогого спектра.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    allBinary = cv2.inRange(hsv, h_min, h_max)

    # Варпим изображение.
    M = cv2.getPerspectiveTransform(polygon, dst)
    warped = cv2.warpPerspective(allBinary, M, (size[1], size[0]), flags=cv2.INTER_LINEAR)
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    # Получаем самые белые столбцы полученного изображения.
    midpoint = histogram.shape[0] // 2
    IndWhitestCoulumnL = np.argmax(histogram[:midpoint])
    IndWhitestCoulumnR = np.argmax(histogram[midpoint:]) + midpoint


    nwindows, window_half = WC, WW

    if nwindows == 0:
        nwindows = 1

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

    # Рисуем линии опредёленных полос на бинаризированном изображении.
    cv2.line(warped, (IndWhitestCoulumnL, 0), (IndWhitestCoulumnL, warped.shape[0]), 355, 3)
    cv2.line(warped, (IndWhitestCoulumnR, 0), (IndWhitestCoulumnR, warped.shape[0]), 355, 3)

    # Высчитываем Координату X центра полосы.
    medium_line_x = (IndWhitestCoulumnL + IndWhitestCoulumnR) // 2

    # Средняя линия второй экран.
    cv2.line(warped, (size[0] // 2, 0), (size[0] // 2, size[1]), 355, 2)

    redlineLen = 8  # от 1 до 10
    greenlineLen = 7

    # Средняя зелёная линия основной экран вывода.
    cv2.line(out_img, (size[0] // 2, int(size[1] // 10 * greenlineLen)), (size[0] // 2, size[1]), (0, 255, 0), 3)

    # Средняя линия полосы.
    cv2.line(out_img, (medium_line_x, int(size[1] // 10 * redlineLen)), (medium_line_x, size[1]), (0, 0, 255), 4)
    cv2.line(out_img, (size[0] // 2, int(size[1] // 10 * redlineLen)), (medium_line_x, int(size[1] // 10 * redlineLen)), (0, 0, 255), 4)

    # Варпим изображение основного экрана.
    out_img = cv2.warpPerspective(out_img, R, (size[1], size[0]), flags=cv2.INTER_LINEAR)

    # Меняем размеры экранов.
    if up_crop == 0:
        up_crop = 1
    if left_crop == 0:
        left_crop = 1
    if down_crop == 0:
        down_crop = 1
    if up_crop == 0:
        up_crop = 1

    refPoint = [(left_crop, up_crop), (right_crop, down_crop)]

    out_img = out_img[up_crop:down_crop, left_crop:right_crop]

    resized = cv2.resize(resized, (330, 330))
    warped = cv2.resize(warped, (330, 330))

    # Выводим все экраны.
    cv2.imshow("result", out_img)
    cv2.imshow("standart", resized)
    cv2.imshow("warped", warped)

cv2.destroyAllWindows()
sys.exit()
