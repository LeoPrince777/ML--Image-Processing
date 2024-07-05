import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
from tensorflow.python.keras.layers import Dropout

pipeline = keras_ocr.pipeline.Pipeline()

removal_list =['md','abdul','ahad','aziz','03','jul','1989','at','04','oct','2015','dkososizico','2025','metroz','dhaka','brtay',]
path = (r'C:\Users\sileo\Downloads\HV_taskDataset\Driv_Li.jpg')
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
box = prediction_groups[0][10]
def inpaint_text(path,removal_list,pipeline):
    
    image = keras_ocr.tools.read(path)
    prediction_groups = pipeline.recognize([image])
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        if box[0] in removal_list:
           x0, y0 = box[1][0]
           x1, y1 = box[1][1] 
           x2, y2 = box[1][2]
           x3, y3 = box[1][3] 
        
           x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
           x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        
           thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
           cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255,thickness)
           image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                 
    return(image)
img = inpaint_text(path, removal_list, pipeline)   
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('text_free_image.jpg',img_rgb)