# Tesseract-opencv-OCR-for-product-labels
The module is able to select a text scene in images containing foreign objects and cut out text paragraphs separately. Unfortunately, the image skew compensation has not yet been implemented.

The module is able to automatically recognize the language, for which it makes a test recognition of text from the cut sample of the image (the crop factor can be adjusted), the language is recognized and re-recognition is done with an explicit indication of the language. If the image contains several text paragraphs in different languages and the language was not specified, the module will automatically recognize the language of each paragraph.

This module implements the calculation of the average number of lines, the average font size and the ratio of the size of the image to the text block in the image. This is required to automatically adjust the filters applied to the image in order to improve the quality of recognition of images with different font sizes, with a different number of lines and different text segmentation.

- Environment used: 
- Python              3.8
- Tesseract           5.0.0-alpha.20200328
- pytesseract         0.3.6
- pycountry           20.7.3
- opencv-python       4.5.1.48
- langdetect          1.0.8
- numpy               1.19.2

If you want to recognize tex in languages other than English, additional language models must be installed in your catalog of trained LSTM Tesseract models. They can be downloaded from the Tesseract repo https://github.com/tesseract-ocr/tessdata
If you are using Windows, then you may need to create the TESSDATA_PREFIX system variable indicating the catalog of LSTM models with additional languages.

If the operating system does not have the Tesseract system variable, then you can manually specify the absolute path to the Tesseract in the file ocr.py.

```
pytesseract.pytesseract.tesseract_cmd = '...absolute_path/tesseract.exe'
```

How use it:
---
Import an image in a format such as jpeg, png.

Create an instance of class -class- and pass an image to it.

Call a method getText() passing it the following parameters:
- **text_lang** (str|boolean) Text language according to ISO 639-3 (alpha-3) standard, or False value if the language should be recognized automatically (automatic recognition slows down text recognition).
- **crop** (boolean)
If you think that the text on the image takes up all the free space and does not contain foreign objects around, set the False flag, if the image is not prepared, select the True flag.
- **set_font** (int)
The parameter sets the target font size in pixels, it is necessary to optimize the quality and speed of recognition. A large font will increase the quality of recognition, but will significantly slow down the calculations, since the image resolution will be increased proportionally.

```
image = cv2.imread('file_directory')
ocr = ImageOCR(image)
recognized_text = ocr.get_text(text_lang='eng', crop=True, set_font=40)
```

How it works:
---

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/1a890c0a7a59aced0baadf4c1c029fb061a33b12/readme_images/preprocessing.png)

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/f0ec47a84e0baebccff35c12dc67d1a6e2e41d21/readme_images/OCR.png)

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/19a6fd5c9823a80d8c86b979d0230dd4f3cac006/readme_images/combine_image.png)
