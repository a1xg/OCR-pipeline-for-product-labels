# Tesseract-opencv-OCR-for-product-labels
- Environment used: 
- Tesseract           5.0.0-alpha.20200328
- pytesseract         0.3.6
- pycountry           20.7.3
- opencv-python       4.5.1.48
- langdetect          1.0.8
- numpy               1.19.2


For multilingual recognition, you need to download the corresponding LSTM language models for Tesseract.


If the image with text contains a lot of unnecessary and needs cropping, set the parameter crop = 1
The recognition quality can be improved by setting the desired font size(set_font parameter) to which the text will be scaled. A font that is too large can slow down the recognition speed.
The language is set in the alpha-3 format or False, if the image language needs to be determined automatically (this will take more time).

If the operating system does not have the Tesseract system variable, then you can manually specify the absolute path to the Tesseract.exe

```
pytesseract.pytesseract.tesseract_cmd = 'D:/Program/Tesseract-OCR/tesseract.exe'
```
```
recognized_text = ocr.getText(text_lang=language, crop=1, set_font=40)
```

How it works:
===========

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/1a890c0a7a59aced0baadf4c1c029fb061a33b12/readme_images/preprocessing.png)

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/f0ec47a84e0baebccff35c12dc67d1a6e2e41d21/readme_images/OCR.png)

![Image alt](https://github.com/a1xg/Tesseract-opencv-OCR-for-product-labels/blob/19a6fd5c9823a80d8c86b979d0230dd4f3cac006/readme_images/combine_image.png)
