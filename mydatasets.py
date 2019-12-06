import re
import os
import random
import tarfile
import urllib
from torchtext import data

class news(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label1_field, label2_field, path, **kwargs):
        """Create an 20news dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label1_field: The field that will be used for high level label data.
            label1_field: The field that will be used for low level label data.
            path: Path to the data file.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label1', label1_field), ('label2', label2_field)]
        examples = []

        with open(path, errors='ignore') as f:
            for line in f:
                bs = line.strip('\n').split('\t')
                if len(bs[0].split()) < 2000:
                    examples.append(data.Example.fromlist(bs[:3], fields))
            print("data size: ", len(examples))
        super(news, self).__init__(examples, fields, **kwargs)