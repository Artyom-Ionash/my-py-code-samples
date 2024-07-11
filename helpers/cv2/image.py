import numpy as np


def add_padding(image, padding=10, color=255):
    """
    Добавляет поля в 10 пикселей к исходному изображению.

    Args:
        image (numpy.ndarray): Исходное изображение.
        padding (int): Размер полей в пикселях (по умолчанию 10).

    Returns:
        numpy.ndarray: Изображение с добавленными полями.
    """
    # Получение размеров исходного изображения
    height, width, _ = image.shape

    # Создание нового холста с увеличенными размерами
    new_height = height + 2 * padding
    new_width = width + 2 * padding
    padded_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * color

    # Копирование исходного изображения в центр нового холста
    x_offset = padding
    y_offset = padding
    padded_image[y_offset : y_offset + height, x_offset : x_offset + width] = image

    return padded_image
