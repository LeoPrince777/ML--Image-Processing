import random 
import os
from PIL import Image
import numpy as np
import cv2
org_img = Image.open(r"C:\Users\sileo\Downloads\HV_taskDataset\Driv_Li.jpg")
ans_dir = "100 Images"
os.makedirs(ans_dir, exist_ok=True)
def hundred_vari(image, count=100):
    width, height = image.size
    for k in range(count):
        img = org_img.copy()

        kernel2 = np.ones((5, 5), np.float32)/25
        kernel = np.ones((5, 5), np.uint8) 

        gaussian = cv2.GaussianBlur(img, (3, 3), 0) 
        medianBlur = cv2.medianBlur(img, 9) 
        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
        img_erosion = cv2.erode(img1, kernel, iterations=1) 
        img_dilation = cv2.dilate(img1, kernel, iterations=1) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = cv2.filter2D(src=org_img, ddepth=-1, kernel=kernel2) 
        img.save(os.path.join(ans_dir, f"image_{k + 1}.jpg"))
hundred_vari(org_img)        
  
