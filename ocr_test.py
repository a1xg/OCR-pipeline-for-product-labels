import time
import os
import re
import glob
import cv2
from ocr import ImageOCR
from postprocessing import TextPostprocessing


def filenameEncoder(dir):
    '''Reads the file number and language(alpha-3 format) of the image from the file name'''
    filename = os.path.basename(dir)
    name = os.path.splitext(filename)[0]
    lang = filename[0:3]
    number = re.findall(r'[0-9]{1,5}', name)[0]

    return (lang, number)


# Get the path to each image in the specified folder
filenames = glob.glob("images_test/*.jpg")
filenames.sort()

for filename in filenames:
    start = time.time()

    language, number = filenameEncoder(filename)
    print(f'Image № {number}, language {language}')

    img = cv2.imread(filename)
    ocr = ImageOCR(img)


    # If the image with text contains a lot of unnecessary and needs cropping, set the parameter crop = 1
    # The recognition quality can be improved by setting the desired font size(set_font parameter) to which the
    # text will be scaled. A font that is too large can slow down the recognition speed.
    # The language is set in the alpha-3 format or False, if the image language needs
    # to be determined automatically (this will take more time).
    recognized_text = ocr.getText(text_lang=False, crop=1, set_font=40)
    print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    print(f"Recognized text:\n{recognized_text}")

    image_with_boxes = ocr.drawBoxes(max_resolution=700)
    cv2.imshow("Selected text", image_with_boxes)
    cv2.waitKey(1000)

    # We iterate over the found text blocks and remove unnecessary characters
    for dict in recognized_text:
        key = list(dict.keys())[0]
        postprocessing = TextPostprocessing()
        cleared_text = postprocessing.stringFilter(input_string=dict[key])
        print(f'Сlear text:\n{cleared_text}')

    print('-' * 30)







