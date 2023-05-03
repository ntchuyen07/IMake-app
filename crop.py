import cv2
import numpy as np

image = cv2.imread('image.jpg')
# brightness adjustment
def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright

cv2.namedWindow("image")
a2 = bright(image, 60)
cv2.imshow('image',a2)
cv2.waitKey(1)
cv2.destroyAllWindows()