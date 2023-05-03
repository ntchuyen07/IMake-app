"""OpenCV Library"""
import cv2
import numpy as np
from PIL.Image import Image
from datetime import datetime

def clamp(pv):
        if pv > 255:
            return 255
        if pv < 0:
            return 0
        else:
            return pv

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
class EffectFilter:
    """
    Class provide default CV2 colormap for apply to image filter
    """

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    @staticmethod
    def cyperpunk_2077(image: Image):
        """
        Apply COLORMAP_PLASMA
        :return: numpy array
        """
        image = np.array(image)
        filtered_image = cv2.edgePreservingFilter(
            image, flags=1, sigma_r=0.6, sigma_s=10
        )
        return filtered_image

    @staticmethod
    def ice(image: Image):
        """
        Apply COLORMAP_OCEAN
        :return: numpy array
        """
        image = np.array(image)
        ice_image = cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
        return ice_image

    @staticmethod
    def snowy(image: Image):
        """
        Apply BRG2GRAY effect
        :return: numpy array
        """
        image = np.array(image)
        # First, convert to grayscale image
        snowy_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        snowy_image_blur = cv2.GaussianBlur(
            snowy_image, (25, 25), 0
        )  # (25, 25) Kernel size
        return cv2.divide(snowy_image, snowy_image_blur, scale=250.0)

    @staticmethod
    def darkness(image: Image):
        """
        Make image darker
        :return: numpy array
        """
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_blur = cv2.GaussianBlur(gray_image, (25, 25), 0)
        darkness_image = cv2.divide(gray_image, gray_image_blur, scale=250.0)

        return cv2.bitwise_not(darkness_image)#reverse pixel value

    @staticmethod
    def gray_nostalgia(image: Image):
        """
        Apply COLORMAP_BONE
        :return: numpy array
        """
        image = np.array(image)
        mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_image = cv2.medianBlur(mask_image, 3)
        nostalgia_image = cv2.applyColorMap(mask_image, cv2.COLORMAP_BONE)

        return nostalgia_image

    @staticmethod
    def cartoon(image: Image):
        image = np.array(image)
        color = cv2.bilateralFilter(image, d=9, sigmaColor=200, sigmaSpace=200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3
        )
        # DF = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
        new_image = cv2.bitwise_and(color, color, mask=edges)
        return new_image

    #Bilateral filtering
    @staticmethod
    def bi_demo(image: Image):  
        image = np.array(image)
        dst = cv2.bilateralFilter(image, 0, 100, 15)
        return dst

    

    #Gaussian noise
    def gaussian_noise(image: Image):  
        image = np.array(image)      
        h, w, c = image.shape
        for row in range(h):
            for col in range(w):
                s = np.random.normal(0, 20, 3)
                b = image[row, col, 0]   #blue
                g = image[row, col, 1]   #green
                r = image[row, col, 2]   #red
                image[row, col, 0] = clamp(b + s[0])
                image[row, col, 1] = clamp(g + s[1])
                image[row, col, 2] = clamp(r + s[2])
        return image
    
    #Sepia_apply
    @staticmethod
    def sepia(image: Image):
        img_sepia = np.array(image, dtype=np.float64) 
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                        [0.349, 0.686, 0.168],
                                        [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
        img_sepia[np.where(img_sepia > 255)] = 255 
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        return img_sepia

    # pencil_apply
    def pencil_sketch_grey(image: Image):
        image = np.array(image) 
        #inbuilt function to create sketch effect in colour and greyscale
        sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        return  sk_color


    # invert_apply
    @staticmethod
    def invert(image: Image):
        img = np.array(image)
        image_invert = cv2.bitwise_not(img)
        return image_invert

    # reverse_apply
    @staticmethod
    def reverse_image(image: Image):
        img = np.array(image)
        igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return 255 - igray

    # threshold_apply
    @staticmethod
    def threshold(image: Image):
        img = np.array(image)
        igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(igray, 127, 255, cv2.THRESH_TOZERO)
        return thresh

    # retro_apply
    def bright(image: Image):
        image = np.array(image)
        img_bright = cv2.convertScaleAbs(image, 0.3)
        return img_bright

    # sharpen adjustment
    def sharpen(image: Image):
        image = np.array(image)
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(image, -1, kernel)
        return img_sharpen

    #HDR effect
    def HDR(image: Image):
        image = np.array(image)
        image_hdr = cv2.detailEnhance(image, sigma_s=6, sigma_r=0.15)
        return image_hdr


    #add text on image filter film
    def add_text(image: Image):
        image = np.array(image)
        now = datetime.now()
        current_time = now.strftime("%d.%m.%Y | %H:%M:%S")
        font=cv2.FONT_HERSHEY_DUPLEX
        image_new = cv2.putText(image,current_time, (100,150), font, 1, (255,255,0), 3, cv2.LINE_AA)
        return image_new


    def vignette(image: Image):
        img = np.array(image)
        rows, cols = img.shape[:2]
        # generating vignette mask using Gaussian kernels
        kernel_x = cv2.getGaussianKernel(cols,450)
        kernel_y = cv2.getGaussianKernel(rows,450)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        output = np.copy(img)

        # applying the mask to each channel in the input image
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask

        return output




    