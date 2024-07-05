from PIL import ImageDraw, Image, ImageFont, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
from tensorflow.keras.layers import Dropout
import random
from faker import Faker
import datetime

pipeline = keras_ocr.pipeline.Pipeline()

removal_list = ['md', 'abdul', 'ahad', 'aziz', '03', 'jul', '1989', 'at', '04', 'oct', '2015', 'dkososizico', '2025', 'metroz', 'dhaka', 'brtay']
path = r'C:\Users\sileo\Downloads\HV_taskDataset\Driv_Li.jpg'

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return x_mid, y_mid

def inpaint_text(path, removal_list, pipeline):
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
        
            thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
            image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    
    return image

img = inpaint_text(path, removal_list, pipeline)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('text_free_image.jpg', img_rgb)

ans_dir = "text_images7c"
os.makedirs(ans_dir, exist_ok=True)

def generate_card(count=100):
    fake = Faker()
    
    for i in range(count):
        template = Image.open('text_free_image.jpg')
        draw = ImageDraw.Draw(template)
        font = ImageFont.load_default(38)
        font1 = ImageFont.load_default(28)
        start = datetime.date(2035, 6, 3)
        end = datetime.date(2035, 6, 30)
        
        records = [
            {
                "Name": fake.name(),
                "Date of Birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d %B %Y"),
                "Fathers Name": fake.name(),
                "Blood Group": fake.random_element(elements=("A+", "B+", "AB+", "O+")),
                "Issue Date": (datetime.date.today() + datetime.timedelta(days=365)).strftime("%d %B %Y"),
                "Date of Validity": fake.date_between(start_date=start, end_date=end).strftime("%d %B %Y"),
                "LNo":fake.bothify(text='??#######?#####'),
                "IA": "BRTA, DHAKA METRO"
            } 
        ]
        
        for record in records:
            draw.text((380, 150), record['Name'], font=font, fill='Green')
            draw.text((380, 220), record['Date of Birth'], font=font, fill='Green')
            draw.text((380, 290), record['Blood Group'], font=font, fill='Green')
            draw.text((380, 360), record['Fathers Name'], font=font, fill='Green')
            draw.text((380, 430), record['Issue Date'], font=font, fill='Green')
            draw.text((672, 428), record['Date of Validity'], font=font, fill='Green')
            draw.text((380, 500), record['LNo'], font=font1, fill='Green')
            draw.text((655, 500), record['IA'], font=font1, fill='Green')
        
        template.save(os.path.join(ans_dir, f"TextAddImage1_{i}.jpg"))

        temp1 = template.resize((597, 308), Image.Resampling.LANCZOS)
        angle = random.randint(0, 360)
        temp1 = temp1.rotate(angle)
        
        background = Image.open(r"C:\Users\sileo\Downloads\HV_taskDataset\BG_IMG.jpg")
        background.paste(temp1, (158, 39, 755, 347))
        background.save(os.path.join(ans_dir, f"BgAddImage_{i}.jpg"))

generate_card()
