import cv2
import numpy as np
import os


def blend_images(big_image_path, small_image_path, output_path):
    # 读取大图像和小图像
    big_image = cv2.imread(big_image_path, cv2.IMREAD_GRAYSCALE)
    small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

    # 确保大图像和小图像都被成功读取
    if big_image is None or small_image is None:
        print("Error: Failed to load images!")
        return

    # 获取大图像和小图像的尺寸
    big_height, big_width = big_image.shape
    small_height, small_width = small_image.shape

    # 创建一个与大图像同样大小的空白图像
    blended_image = big_image.copy()

    # 计算小图像应该嵌入的位置，开始的坐标是大图像的1/5处
    start_x = big_width // 5
    start_y = big_height // 5

    # 遍历大图像中的像素，融合小图像的像素
    for i in range(small_height):
        for j in range(small_width):
            # 计算当前像素点在大图像中的对应位置
            big_i = start_y + i
            big_j = start_x + j

            # 确保在大图像范围内
            if big_i < big_height and big_j < big_width:
                big_pixel_value = big_image[big_i, big_j]
                small_pixel_value = small_image[i, j]

                # 根据灰度值条件进行融合
                if big_pixel_value == small_pixel_value:
                    blended_value = big_pixel_value
                # elif big_pixel_value < 80 and small_pixel_value < 80:
                #     blended_value = (big_pixel_value + small_pixel_value) // 2
                # elif small_pixel_value > 50 and big_pixel_value < 40:
                elif small_pixel_value > 20:
                    blended_value = small_pixel_value

                else:
                    blended_value = big_pixel_value

                # 将融合后的值赋值给新的图像
                blended_image[big_i, big_j] = blended_value

    # 保存融合后的图像
    cv2.imwrite(output_path, blended_image)
    print(f"Blended image saved to {output_path}")