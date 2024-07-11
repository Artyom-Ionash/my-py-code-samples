import cv2
from typing import Literal, Sequence, Union, Tuple, List

import numpy as np

# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

ContourFeature = Literal["point", "width", "height", "area", "perimeter", "solidity", "squareness", "aspect_ratio", "circularity"]
white = (255, 255, 255)


def diagonal_regeneration(target_contour: np.ndarray):
    x, y, w, h = cv2.boundingRect(target_contour)

    convected_counter = convex(target_contour)

    # Вычисление центра бокса
    center_x = x + w // 2
    center_y = y + h // 2

    # Создание матрицы поворота на 180 градусов
    M = cv2.getRotationMatrix2D((center_x, center_y), 180, 1.0)

    # Поворот контура
    rotated_contour = cv2.transform(convected_counter, M).squeeze()
    regenerated_contour = convex(np.concatenate([rotated_contour, target_contour], axis=0))

    return regenerated_contour


def convex(target_contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(target_contour)

    # Создание нового холста с зазором в 5 пикселей
    new_width = x + w + 5
    new_height = y + h + 5
    canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Перебор контуров и рисование их с использованием cv2.convexHull()
    hull = cv2.convexHull(target_contour)
    cv2.drawContours(canvas, [hull], 0, white, 2)

    mask: np.ndarray = cv2.inRange(canvas, white, white)
    counters, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return counters[0]


def extract_features(contour: np.ndarray, feature_names: Sequence[ContourFeature] = ()):
    row: dict[ContourFeature, Union[int, float, Tuple[int, int]]] = {}

    x, y, width, height = cv2.boundingRect(contour)
    row["point"] = (x, y)
    row["width"] = width
    row["height"] = height

    if any(f in feature_names for f in ("area", "circularity", "solidity", "squareness")):
        row["area"] = cv2.contourArea(contour)
    if any(f in feature_names for f in ("perimeter", "circularity", "squareness")):
        row["perimeter"] = cv2.arcLength(contour, True)

    if "aspect_ratio" in feature_names:
        row["aspect_ratio"] = float(row["width"]) / row["height"]
    if "circularity" in feature_names:
        row["circularity"] = (row["perimeter"] ** 2) / (4 * np.pi * row["area"])
    if "solidity" in feature_names:
        row["solidity"] = row["area"] / (row["width"] * row["height"])
    if "squareness" in feature_names:
        row["squareness"] = row["perimeter"] / np.sqrt(row["area"])

    return row


def color_split(image: np.ndarray, tolerance=3):
    colors_contours: List[Tuple[np.ndarray]] = []

    # Конвертация в HSV-цветовое пространство
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение уникальных цветов, исключая белый
    unique_colors: np.ndarray = np.unique(hsv_image.reshape(-1, hsv_image.shape[2]), axis=0)
    unique_colors: np.ndarray = unique_colors[~np.all(unique_colors == [0, 0, 255], axis=1)]
    # Создание структурирующего элемента для "дилатации"

    kernel = np.ones((tolerance, tolerance), np.uint8)

    # Создание двоичных масок и извлечение контуров цветов
    for i, color in enumerate(unique_colors):
        mask: np.ndarray = cv2.inRange(hsv_image, color, color)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colors_contours.append(contours)

    return colors_contours


def save_to_png(contour: np.ndarray, file_path: str):
    """
    Сохраняет контур в файл в формате PNG.

    Args:
    contour (numpy.ndarray): контур для сохранения
    file_path (str): путь к файлу для сохранения
    """
    # Определяем размер изображения, достаточный для контура
    x, y, w, h = cv2.boundingRect(contour)
    image_size = (w + 50, h + 50)

    # Создаем пустое изображение для отрисовки контура
    image = np.zeros(image_size, dtype=np.uint8)
    image.fill(255)  # Заливаем белым цветом

    # Отрисовываем контур на изображении
    cv2.drawContours(image, [contour], 0, (128, 128, 0), 1)

    # Сохраняем изображение в PNG-формате
    cv2.imwrite(file_path, image)

    print(f"Контур сохранен в файл: {file_path}")
