import re
import base64
import numpy as np
import cv2
import pytesseract
import pycountry
from langdetect import detect, DetectorFactory
from .ocr_settings import *
# Absolute path to tesseract.exe file if environment variable is not working correctly
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# The module is able to select a text scene in images containing foreign objects
# and cut out text paragraphs separately. Unfortunately, the image skew compensation
# has not yet been implemented.
#
# The module is able to automatically recognize the language, for which it makes a
# test recognition of text from the cut sample of the image (the crop factor can be adjusted),
# the language is recognized and re-recognition is done with an explicit indication of the language.
# If the image contains several text paragraphs in different languages and the
# language was not specified, the module will automatically recognize the language of each paragraph.
#
# This module implements the calculation of the average number of lines, the average font size
# and the ratio of the size of the image to the text block in the image. This is required to
# automatically adjust the filters applied to the image in order to improve the quality of
# recognition of images with different font sizes, with a different number of lines and
# different text segmentation.

# TODO
#  Find ways to process multiple images simultaneously.
#  Check the recognized language in the list of available Tesseract otherwise return False.
#  Process langdetect exception in case of image without text.
#  Transfer the ocr module to the client side and rewrite it to JavaScript accordingly.

class ImageOCR:
    def __init__(self, img):
        self.original = img
        self.img = self._decode_image(img)
        self.result = [] # output list of dictionaries with recognized text in the format {'lang':'text'}
        self._boxes = []

    def _decode_image(self, input_img) -> np.ndarray:
        '''Decode bytes image to numpy format'''
        decoded_img = cv2.imdecode(np.fromstring(input_img, np.uint8), cv2.IMREAD_UNCHANGED)
        return decoded_img

    def _encode_image(self, input_img:np.ndarray):
        '''Encodes an np.ndarray into a base64 string'''
        retval, buffer = cv2.imencode('output.jpg', input_img)
        # Convert to base64 encoding and show start of data
        base64_string = base64.b64encode(buffer)
        output_image = {
            'height':input_img.shape[0],
            'width':input_img.shape[1],
            'image':base64.b64encode(buffer),
        }
        return output_image

    def _image_preprocessing(self) -> np.ndarray:
        '''Flattening the image histogram'''
        grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize= (8,8))
        normalised_img = clahe.apply(grayscale)
        self.img = normalised_img

    def _resize_image(self, image:np.ndarray, max_resolution:int) -> np.ndarray:
        """ The method resizes the image to the maximum allowed
        maintaining the original proportions regardless of vertical or
        horizontal orientation, for example: from image 2000x1600px or 1600x2000px, if the maximum dimension
        set as 1000 pixels, then 1000x800 or 800x1000 can be obtained.
        """
        max_dim = max(image.shape[0:2])
        scale_coef = max_dim / max_resolution
        new_dim = (int(image.shape[1] / scale_coef), int(image.shape[0] / scale_coef))
        img_scaled = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
        return img_scaled

    def _measure_strings(self) -> tuple:
        '''Method of counting text lines in an image and measuring the average font height
        :param image:
        :return:
        '''
        num_lines = []
        font_size = []

        height, width = self.img.shape[0:2]
        for slice in MEASURE_STRINGS_SLICES:
            newX = width*slice[0]
            newW = width*slice[1] - width*slice[0]
            crop = self.img[0:height, int(newX):int(newX+newW)]
            crop = self._get_text_mask(crop, font_size=0, num_lines=0)

            # Reduce the 2D array along the X axis to a 1D array
            hist = cv2.reduce(crop, 1, cv2.REDUCE_AVG).reshape(-1)
            H, W = crop.shape[:2]
            lines = [y for y in range(H - 2) if hist[y] <= MEASURE_TH and hist[y + 1] > MEASURE_TH]
            if len(lines) > 0:
                font_size.extend([lines[i+1] - lines[i] for i in range(len(lines) - 1)])
                num_lines.append(len(lines))
        # Calculate the average line height in pixels
        mean_font_size = int(np.array(font_size).mean()) if len(font_size) > 1 else 0
        # Calculate the average number of text lines
        mean_num_lines = int(np.array(num_lines).mean()) if len(num_lines) > 1 else 0

        return (mean_font_size, mean_num_lines)

    def _get_text_mask(self, image:np.ndarray, font_size:int, num_lines:int) -> np.ndarray:
        """
        The method searches the image for a text block and creates
        mask to use it to find outlines bounding the text and crop the excess part of the image.
        """
        rect_kernel_size = (100, 5)  # (W, H) default values 100, 5
        sq_kernel_size = (100, 5)  # (W, H) default values 100, 5
        if bool(font_size) == True:
            # calculate the coefficient of sparseness of the text in the image for optimal filter settings
            h_text_coef = font_size*num_lines/image.shape[0]
            rect_kernel_size = (int(font_size*0.6), int(font_size*0.4/h_text_coef))
            sq_kernel_size = (int(font_size*0.5), int(font_size*0.4/h_text_coef))
        elif bool(font_size) == False:
            None
        # Apply filters to the image
        imgBlur = cv2.GaussianBlur(image, (3, 3), 0) # 3Х3
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, rect_kernel_size)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, sq_kernel_size)
        blackHat = cv2.morphologyEx(imgBlur, cv2.MORPH_BLACKHAT, kernel1)
        gradX = cv2.Sobel(blackHat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel1)
        threshold1 = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        threshold2 = cv2.morphologyEx(threshold1, cv2.MORPH_CLOSE, kernel2)
        text_mask = cv2.erode(threshold2, None, iterations=4)

        # remove the edge of the image, so that in the future there
        # would be no outlines recognition outside the image
        border = int(image.shape[1] * 0.05)
        text_mask[:, 0: border] = 0
        text_mask[:, image.shape[1] - border:] = 0

        return text_mask

    def _get_binary_images(self, image:np.ndarray, font_size) -> list:
        '''The method crops the text area of interest in the image, brings the
        resolution of the cropped area to the new standard maximum resolution.
        '''
        binary_images = []
        # If the list of bounding boxes is empty, then the entire image will be the bounding box
        self._boxes = [[0, 0, image.shape[1], image.shape[0]]] if len(self._boxes) <= 0 else self._boxes
        # Cut out blocks with text from the original image using bounding boxes
        for index, box in enumerate(self._boxes):
            (x, y, w, h) = box
            cropped_img = image[y:y + h, x:x + w]
            num_lines = int(cropped_img.shape[0]/font_size) # count the number of lines in the image
            # If the number of lines is more than the threshold value, then we process the image, otherwise we skip
            if num_lines > LINE_NUM_THRESHOLD:
                # blur the image
                blur_size = int(np.ceil(font_size*0.2))
                blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1
                gaussian_blured = cv2.GaussianBlur(cropped_img, (blur_size, blur_size), 10)
                # sharpen the image
                test_sharpened = cv2.addWeighted(cropped_img, 1.5, gaussian_blured, -0.9, 0)
                # blur the image in order to minimize noise in the image in the next step
                median_size = int(np.ceil(font_size*0.02))
                median_size = median_size if median_size % 2 != 0 else median_size + 1
                medianblur = cv2.medianBlur(test_sharpened, median_size) # default 3
                # image binarization
                test_thresh = cv2.threshold(medianblur, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
                # We calculate the percentage of black pixels relative to the total number of pixels
                percent_zeros = int(cv2.countNonZero(test_thresh)*100/(test_thresh.shape[1]*test_thresh.shape[0]))
                # If the background of the picture is more than n% black, and the text is white, then invert the colors
                img_binary = cv2.bitwise_not(test_thresh) if percent_zeros < 40 else test_thresh

                # We calculate the ratio of the height of the image to the font and the number of lines
                scale_coef = font_size/TARGET_FONT_SIZE
                max_dimension = max(img_binary.shape[0:2])
                new_max_resolution = max_dimension/scale_coef
                # Resize excessively large images based on the desired font size
                scaled_binary_img = self._resize_image(img_binary, new_max_resolution)
                binary_images.append((index,scaled_binary_img))
        # return an array of binary images, if any.
        return binary_images if len(binary_images) > 0 else False

    def _find_boxes(self, mask_img:np.ndarray) -> None:
        """
        The method accepts a binary image mask that selects text blocks.
        The quantity parameter is intended to limit the number of boxes, remove duplicates and very small boxes.
        The method will return the maximum {quantity} of boxes with the maximum area.
        """
        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = mask_img.shape[0]*mask_img.shape[1]
        boxes = []
        for cont in contours:
            # remove boxes whose area is less than the threshold value
            (x, y, w, h) = cv2.boundingRect(cont)
            if w*h < img_area/200:
                continue
            else:
                # Expand the remaining boxes (%) using the coefficients by, bx so
                # that the text is guaranteed to fit in them
                by = int(h*TEXT_BOX_GAP_Y/100)
                bx = int(w*TEXT_BOX_GAP_X/100)
                x1, x2 = abs(x - bx), (x + w + bx)
                y1, y2 = abs(y - by), (y + h + by)
                # We check if the coordinates x2, y2 go beyond the image and if they do,
                # then the coordinates are equal to the maximum dimension of the image
                if x2 > mask_img.shape[1]:
                    x2 = mask_img.shape[1]
                if y2 > mask_img.shape[0]:
                    y2 = mask_img.shape[0]

                new_W, new_H = (x2 - x1), (y2 - y1)

                boxes.append([x1, y1, new_W, new_H])

        self._boxes = self._find_largest_boxes(boxes=boxes) if len(boxes) > NUM_OF_LARGEST_BOXES else boxes

    def _find_largest_boxes(self, boxes:np.array) -> list:
        '''The method takes a list of all found boxes and returns the given
        number (not more than: NUM_OF_LARGEST_BOXES :) of boxes with the largest area'''
        boxes_array = np.array(boxes)
        areas = np.prod(boxes_array[:,2:4], axis=1)
        max_areas_indises = np.argpartition(areas, -NUM_OF_LARGEST_BOXES)[-NUM_OF_LARGEST_BOXES:]
        bigest_boxes = [boxes_array[i].tolist() for i in max_areas_indises]
        return bigest_boxes

    def _crop_image(self, image:np.ndarray) -> np.ndarray:
        """The method crops the image proportionally to the crop ratio (floating point numbers from 0 to 1)
        relative to the center of the image."""
        x, y, h, w = 0, 0, image.shape[0], image.shape[1]
        # For correct language recognition, the width is additionally multiplied by 1.5 (you can experiment)
        new_w, new_h = (w*CROP_FACTOR*1.5), (h*CROP_FACTOR)
        new_x, new_y = (w - new_w), (h - new_h)
        cropped_img = image[int(new_y):int(new_y+new_h), int(new_x):int(new_x+new_w)]
        return cropped_img

    def _recognize_text(self, conf:str, image:np.ndarray) -> str:
        """Text recognition method"""
        return pytesseract.image_to_string(image, config=conf)

    def _detect_lang(self, text:str) -> str:
        # remove all non-alphanumeric characters
        cleared_text = re.sub(r'[\W_0-9]+', ' ', text)
        DetectorFactory.seed = 0
        alpha_2_lang = detect(cleared_text.lower())
        # Convert ISO 639-3 language code format from alpha_2 to alpha_3
        langdict = pycountry.languages.get(alpha_2=alpha_2_lang)
        alpha_3_lang_code = langdict.alpha_3
        return alpha_3_lang_code

    def get_text(self, text_lang:str, crop:bool) -> list:
        """text_lang The language must be specified in alpha-3 format,
        if the language is unknown, then the text_lang parameter must be set to False.
        If the language is not specified, then it will be recognized automatically,
        but it will take more time, since text recognition needs to be done twice.
        The crop parameter is set to True if the text needs to be cropped,
         and False if the block of text has already been cut from the photo.
         """
        self._image_preprocessing()
        font_size, num_lines = self._measure_strings()
        mask_img = self._get_text_mask(self.img, font_size, num_lines)
        if crop == True:
            self._find_boxes(mask_img=mask_img)
            binary_images = self._get_binary_images(self.img, font_size)
        else:
            binary_images = self._get_binary_images(self.img, font_size)
        if binary_images == False:
            return False
        # Loop through images prepared for OCR
        for index, image in binary_images:
            if text_lang == False:
                # a cropped sample image is used to speed up language recognition.
                sample_image = self._crop_image(image)
                multilang_recog_text = self._recognize_text(TESSERACT_DEFAULT_CONFIG, sample_image)
                # Detect language
                recognized_lang = self._detect_lang(multilang_recog_text)
                # After the exact definition of the language, we make repeated
                # recognition with the exact indication of the language
                custom_config = (f'-l {recognized_lang} --oem 1 --psm 6')
                ocr_result = self._recognize_text(custom_config, image)
                self.result.append({
                    'lang':recognized_lang,
                    'box_index':index,
                    'text': ocr_result
                })
            else:
                # Recognition option if the language of the text is known
                config = (f'-l {text_lang} --oem 1 --psm 6')
                ocr_result = self._recognize_text(config, image)
                self.result.append({
                    'lang': text_lang,
                    'box_index': index,
                    'text': ocr_result
                })
        return self.result

    def draw_boxes(self, **kwargs) -> np.ndarray:
        """
        Method for drawing bounding rectangles
        It is intended for debugging of the module or delivery to the
        user of the image with the selected text

        :param index: (int) Index of the box text in the array self._boxes,
        optional parameter to print one selected box.
        :param max_resolution: (int) An optional parameter to scale the image
        to a specified maximum measurement of height or width.
        :param color: (tuple) color RGB for example (255,122,4)
        :param thickness: (int) line thickness(pixels) of box
        :param bytes: (bool) When specifying the parameter, truth will return
        a byte-image, by default it will return an nd.array
        """
        img = self._decode_image(self.original)

        boxes = self._boxes # default value - all boxes
        if 'index' in kwargs:
            index = kwargs['index']
            boxes = [self._boxes[index]]

        line_color = (255, 0, 0) # default
        if 'color' in kwargs:
            line_color = kwargs['color']

        line_thickness = 10 # default
        if 'thickness' in kwargs:
            line_thickness = kwargs['thickness']

        for box in boxes:
            (x,y,w,h) = box
            cv2.rectangle(img, (x, y), ((x + w), (y + h)), line_color, line_thickness)

        # If the maximum resolution is set, then the picture will be changed, otherwise return the original
        if 'max_resolution' in kwargs:
            img = self._resize_image(img, kwargs['max_resolution'])

        if kwargs.get('base64') == True:
             return self._encode_image(img)

        return img
