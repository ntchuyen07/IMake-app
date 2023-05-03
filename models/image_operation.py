""" PIL module """
import re
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
import random


class ImageOperation:
    """
    Class hold Image object and Image Manipulation
    """

    @staticmethod
    def get_image_array(img: Image):
        return np.array(img)

    @staticmethod
    def get_information(img: Image) -> dict:
        """
        Get basic information of Image
        :return: dict contain image's information : format, size (), mode
        """
        return {
            "name": img.filename,
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
        }

    @staticmethod
    def resize_image(img: Image, radius: int) -> Image:
        """
        Resize image (Create low resolution image from original image)
        :param img: Image
        :param radius: how many times decrease the resolution
        :return: Image object (PIL)
        """
        return img.resize(
            img.width // radius,
            img.height // radius,
            resample=Image.Resampling.HAMMING,  # Better performance and quality
        )

    @staticmethod
    def transpose_image(img: Image, direction: Image.Transpose):
        """
        Transpose image to direction
        :return: new Image object (PIL)
        """
        return img.transpose(direction)

    @staticmethod
    def rotate_image(img: Image, degrees: int) -> Image:
        """
        Rotate image by given degrees
        :param img: Image
        :param degrees: int
        :return: new Image object (PIL)
        """
        return img.rotate(degrees, expand=True)

    @staticmethod
    def blur_image(image: Image, radius: float):
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def brightness_image(img: Image, factor: float) -> Image:
        """
        Adjust image Brightness
        :param img: Image
        :param factor: > 1: Brighten image
        :param factor: 0 < x < 1: Darkened image
        :return: new Image object (PIL)
        """
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    @staticmethod
    def color_image(img: Image, factor: float) -> Image:
        """
        Adjust image Color
        :param img: Image
        :param factor: int
        :return: new Image object (PIL)
        """
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    @staticmethod
    def sharpen_image(img: Image, factor: float) -> Image:
        """
        Adjust image sharpness
        :param img: Image
        :param factor: int
        :return: new Image object (PIL)
        """
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)


    @staticmethod
    def contrast_image(img: Image, factor: int) -> Image:
        """
        Adjust image Contrast
        :param img: Image
        :param factor: int
        :return: new Contrast image object (PIL)
        """
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    @staticmethod
    def dilate_image(image: Image, cycle: int) -> Image:
        """
        Image Dilation
        :param cycle: cycles of dilation
        :param image: Image
        :return: Image object (PIL)
        """
        for _ in range(cycle):
            image = image.filter(ImageFilter.MaxFilter(size=3))
        return image

    @staticmethod
    def erode_image(img: Image, cycle: int) -> Image:
        """
        Image Erosion
        :param cycle: cycles of erosion
        :param img: Image
        :return: Image object (PIL)
        """
        for _ in range(cycle):
            img = img.filter(ImageFilter.MinFilter(size=3))
        return img

    @staticmethod
    def convert_to_sketch_image(img: Image) -> Image:
        """
        Convert to Sketch image
        :return: Image object (PIL)
        """
        img_gray_smooth = img.filter(ImageFilter.SMOOTH)
        edge_smooth = img_gray_smooth.filter(ImageFilter.FIND_EDGES)

        # Get bands[0]: Black and white for better result
        bands = edge_smooth.split()

        # Invert the color
        outline = bands[0].point(lambda x: 255 - x)
        return outline

    @staticmethod
    def invert_image(img: Image) -> Image:
        """
        Return the invert version of image
        :return: Image
        """
        if img.mode == "RGBA":
            img = img.convert("RGB")

        return ImageOps.invert(img)

    @staticmethod
    def gamma_correction(img: Image, gamma) -> Image:
        normalization_const = 255.0 / np.float_power(255, gamma)

        image = np.array(img)
        # compute output = constant * in^gamma
        output = np.uint8(normalization_const * np.float_power(image, gamma))

        return Image.fromarray(output)

    @staticmethod
    def histogram_equalization(img: Image) -> Image:
        """
        Histogram Equalization using Pillow Library
        :param img: Image
        :return: Image
        """
        img = img.convert("L")
        input_image = ImageOperation.get_image_array(img)
        image = cv2.equalizeHist(input_image)
        # return ImageOps.equalize(img)
        return Image.fromarray(image)

    @staticmethod
    def log_transform(img: Image) -> Image:
        if img.mode == "RGBA":
            img = img.convert("RGB")

        image = np.array(img)
        normalization_const = 255 / np.log(1 + np.max(image))
        log_img = normalization_const * (np.log(image + 1))
        # convert to int
        log_img = np.array(log_img, dtype=np.uint8)

        return Image.fromarray(log_img)

    @staticmethod
    def gamma_transform(img: Image, gamma_value: float):
        image = np.array(img)
        normalization_const = 255.0 / np.float_power(255, gamma_value)
        gamma_img = np.uint8(normalization_const * np.float_power(image, gamma_value))

        return Image.fromarray(gamma_img)
