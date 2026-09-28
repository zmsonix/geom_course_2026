import numpy as np

def pad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.pad(image, pad, mode="constant", constant_values=0)
    elif mode == "edge":
        return np.pad(image, pad, mode="edge")
    else:
        raise ValueError("ошибка")


def box_blur(image: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size должен быть нечётным числом")

    pad = kernel_size // 2
    padded = pad_image(image, pad, mode).astype(np.float64)
    result = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = window.mean()

    result = np.clip(np.floor(result + 0.5), 0, 255).astype(np.uint8)
    return result