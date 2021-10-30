DEFAULT_LANG = 'eng'

# The target font size(pix) to which the text will be scaled, this font size will
# be used for Tesseract recognition. Large size improves accuracy,
# but requires more time and computational resources. (int)
TARGET_FONT_SIZE = 40

# Windows Tesseract path
TESSERACT_PATH = 'D:/Program/Tesseract-OCR/tesseract.exe'

# text box border gap(%).
TEXT_BOX_GAP_X = 5
TEXT_BOX_GAP_Y = 5

# Set the threshold for the number of lines in the image,
# images with fewer lines will be deleted.(int)
LINE_NUM_THRESHOLD = 2

# slices of the image along the X axis relative to the width of the image,
# to enhance counting accuracy for images with distorted perspective or
# with imperfect horizontal lines. values: tuples like ((0...X, X...1),...)
MEASURE_STRINGS_SLICES = ((0.35, 0.45), (0.5, 0.6), (0.65, 0.75))

#A cut-off threshold for counting the number of white and black sequences in the
# array and determining what we count as white and what as black. Accepts values from 1 to 255.
#For example, in the array [100,100,5,3,6,10,0,4,122,150,255,0,4,2,110,100,90]
# with MEASURE_TH = 10, the white and black areas are switched at indices 7 and 13.
MEASURE_TH = 2

# If more than NUM_OF_LARGEST_BOXES text boxes are found in the picture, only
# NUM_OF_LARGEST_BOXES boxes with the largest area will be saved. (int)
NUM_OF_LARGEST_BOXES = 3

# Crop factor for cropping a sample text image and trial OCR and
# language detection, required to optimize the speed of two-pass OCR:
# 1) multilingual inaccurate recognition on a sample of text.
# 2) accurate recognition on the entire text with the selected language. (float) values [0...1].
CROP_FACTOR = 0.6

# If the image language is unknown, then the tesseract will
# make primary recognition in several languages, low quality
TESSERACT_DEFAULT_CONFIG = '-l rus+deu+eng --oem 1 --psm 6'

'''
To deploy to Heroku, you don't need to configure pytesseract.tesseract_cmd and you just need to
add environment variable:
Key: tesseract 
Value: ./.apt/usr/bin/tesseract
'''