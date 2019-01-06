from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw

import cv2

def cropImage(inputImageFile, cropCordinates, outputImageFile = './media/cropped/croppedImage.jpg'):
    if len(cropCordinates) != 4:
        return False
    else:
        # read image

        img = cv2.imread(inputImageFile)

        # crop image
        x1, y1, x2, y2 = cropCordinates
        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]

        # save cropped image
        cv2.imwrite(outputImageFile, cropped_image)

        return outputImageFile

def read_color(input_filename):

    client=vision.ImageAnnotatorClient()

    with open(input_filename, 'rb') as image:
        image = types.Image(content=image.read())
        results = client.image_properties(image=image)
        colors = results.image_properties_annotation.dominant_colors.colors
        main_colors=[]
        for each_color in colors:
            if each_color.score > 0.1:
                main_colors.append(each_color)
    return (main_colors[0].color.red, main_colors[0].color.green, main_colors[0].color.blue)

def makeColorBox(color_R,color_G,color_B,outputImagePath = './media/colorbox/colorbox.jpg'):
    im = Image.open(outputImagePath)

    draw = ImageDraw.Draw(im)
    draw.rectangle(xy=(20, 20), fill=(color_R,color_G,color_B))

    ImageDraw
    return True