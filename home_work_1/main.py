import numpy as np

from blur import pad_image, box_blur


def main():
    np.random.seed(42)

    image = np.random.randint(1, 11, size=(10, 8), dtype=np.uint8)

    print("Исходное изображение (10x8):")
    print(image)

    print("\npad_image, mode='zero', pad=1:")
    print(pad_image(image, 1, "zero"))

    print("\npad_image, mode='edge', pad=1:")
    print(pad_image(image, 1, "edge"))

    print("\nbox_blur, kernel_size=3, mode='zero':")
    print(box_blur(image, 3, "zero"))

    print("\nbox_blur, kernel_size=3, mode='edge':")
    print(box_blur(image, 3, "edge"))


if __name__ == "__main__":
    main()