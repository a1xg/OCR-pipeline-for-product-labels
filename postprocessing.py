import re

class TextPostprocessing:
    def __init__(self):
        self._output_string = str

    def stringFilter(self, input_string):
        '''Cleaning text'''
        # cut out word break with line break
        string = re.sub(r'-\n', '', input_string)
        # replace the usual line break with a space
        string = re.sub(r'\n', ' ', string)
        #  Filter replaces parentheses with commas
        string = re.sub(r'[\{\[\(\)\]\}\|]', ',', string)
        # The filter replaces special characters with commas
        string = re.sub(r'[*«»+=~<>:;.&]', ',', string)
        # Percentage filter
        string = re.sub(r'\d*\.?,?\s?\d?\s?%', ',', string)
        # The filter strips off escape characters and word breaks with line breaks
        string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ',', string)
        # The filter replaces consecutive commas with spaces if there is more than one
        string = re.sub(r'(,\s+|,+){2,}', ',', string)
        # Cut out the quotes
        string = re.sub(r'[\'\"]', '', string)
        # remove the combination of spaces and commas
        string = re.sub(r'(\s+,|,\s+)', ',', string)
        # Replace multiple spaces with one space
        self._output_string = re.sub(r'\s+', ' ', string)

        return  self._output_string.lower()