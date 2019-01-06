from PIL import Image, ImageDraw
import os

def process(img):
    img = Image.open(img)
    d = ImageDraw.Draw(img)
    d.text((150,150), "This is Result Image", fill=(255,0,0))
    return img

