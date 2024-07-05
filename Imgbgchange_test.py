from PIL import Image, ImageDraw
img =Image.open(r" C:\Users\sileo\Downloads\HV_taskDataset\Driv_Li ")
img1 = img.convert("RGB")
seed =(263, 70)
rep_value =(255,255,0)
ImageDraw.floodfill(img,seed,rep_value,thresh=50)
img.show