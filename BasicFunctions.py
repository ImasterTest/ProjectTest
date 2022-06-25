import io
import os
import xml.etree.ElementTree as ET
import zipfile
from glob import glob
from tkinter import filedialog

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

import PrintUtils

"""
This is a utils functions class.
It has functions for different parts of the project
"""

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

      Puts image into numpy array.

    :param path: path: a file path (this can be local or on colossus)
    :return: numpy array with shape (img_height, img_width, 3) in RGB format
    """
    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    # image = Image.open(BytesIO(img_data))
    # (im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)

    image = Image.open(path)
    image = image.convert('RGB')
    return np.array(image)

def ask_directory(force=True, **kwargs):
    """
    Asks the user to select a directory.

    :param force: Force the user to choose a valid directory
    :param kwargs: arguments for the ask_directory
    :return: The selected directory path
    """
    output_dir = filedialog.askdirectory(**kwargs)
    while (output_dir == '' or output_dir is None) and force:
        PrintUtils.error('You must choose a directory', show_time=False)
        output_dir = filedialog.askdirectory(**kwargs)
    return output_dir

def ask_empty_directory(force=True, **kwargs):
    """
    Asks the user to select an empty directory.

    :param force: Force the user to choose a valid directory
    :param kwargs: arguments for the ask_directory:
                initialdir: The initial directory to open in dialog
                title: The title of the file dialog window
    :return: The selected directory path
    """
    # root.focus_force()
    dire = ask_directory(force, **kwargs)
    if not force and (dire == '' or dire is None):
        return dire
    while (dire == '' or dire is None) or not len(os.listdir(dire)) == 0:
        PrintUtils.error('The directory must be empty', show_time=False)
        dire = ask_directory(force, **kwargs)
        if not force and (dire == '' or dire is None):
            return dire
    return dire


def ask_for_file(force=True, **kwargs):
    """
    Asks the user to select a file.

    :param force: Force the user to choose a valid file
    :param kwargs: arguments for the askopenfilename:
                initialdir: The initial directory to open in dialog
                title: The title of the file dialog window
                filetypes: The accepted filetypes to select
    :return: The path to the selected file
    """
    # root.focus_force()
    # root = Tk()
    output = filedialog.askopenfilename(**kwargs)
    # root.destroy()
    while (output is None or output == '') and force:
        PrintUtils.error('You must choose a file', show_time=False)
        # root = Tk()
        output = filedialog.askopenfilename(**kwargs)
        # root.destroy()
    return output

def ask_save_file(force=True, **kwargs):
    """
        Asks the user to select a file to save.

        :param force: Force the user to choose a valid directory
        :param kwargs: arguments for the askopenfilename:
                    initialdir: The initial directory to open in dialog
                    title: The title of the file dialog window
                    filetypes: The accepted filetypes to select
                    defaultextension: The accepted extension on the filetype
        :return: The path to the selected file
    """
    # root.focus_force()
    # root = Tk()
    output = filedialog.asksaveasfilename(**kwargs)
    # root.destroy()
    while output is None and force:
        PrintUtils.error('You must choose a file', show_time=False)
        # root = Tk()
        output = filedialog.asksaveasfilename(**kwargs)
        # root.destroy()
    return output


def download_file(url, output_path):
    """
    Downloads a file from the given url to the given path.

    :param url: The url to download from
    :param output_path: The output path for the downloaded file
    :return: None
    """
    r = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(r.content)


def extract_zip(path, output_path):
    """
    Extracts a zip to the given output_path.

    :param path: The path to the zip file
    :param output_path: The path to the extract direcory
    :return: True if successful, False otherwise
    """
    ext = path.split('.')[-1]
    if ext == 'zip':
        PrintUtils.info('Got a zip file, extracting...')
        with zipfile.ZipFile(path, 'r') as zip_f:
            zip_f.extractall(output_path)
        PrintUtils.info(f'Finished extracting')
        return True
    else:
        PrintUtils.error('The file is not a zip file')
        return False

def to_zip(path, output_file):
    """
    Convert a directory to a zip file.

    :param path: The path to the directory
    :param output_file: The path to the outputted zip file
    :return: None
    """
    with zipfile.ZipFile(output_file, 'w') as zip_f:
        for root, dirs, files in os.walk(path):
            for f in files:
                zip_f.write(os.path.join(root, f),
                            # os.path.relpath(os.path.join(root, f), os.path.join(path, '..')))
                            os.path.relpath(os.path.join(root, f), path))

def xml_to_csv(path, classes_filter):
    """
    Takes all the xml files from the path and converts them to a Pandas dataframe.

    :param path: Path to the directory containing the xml files
    :param classes_filter: Filter for the classes in the labels files
    :return: A Pandas dataframe with the given columns: [image_name, image_width, image_height, class, xmin, ymin, xmax, ymax]
             Note: There is a row created for each label, if a file has more than one label
                   it will create a row for each one
             The names of the classes found in the xml files
    """

    images_extension = 'jpg'

    classes_names = []  # The names of the existing classes names in the dataset
    xml_list = []  # Every label in every file in format ['filename', 'image_width', 'image_height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    for xml_file in glob(os.path.join(path + '/*.xml')):
        xml_tree = ET.parse(xml_file)  # The xml tree
        xml_root = xml_tree.getroot()  # The root element of the xml file

        filename = xml_root.find('filename').text + '.' + images_extension  # The image filename element
        image_width = int(xml_root.find('size').find('width').text)  # The image's width size element
        image_height = int(xml_root.find('size').find('height').text)  # The image's height size element
        for obj in xml_root.findall('object'):
            clazz = obj.find('name').text  # The object's class element
            xmin = int(obj.find('bndbox').find('xmin').text)  # The object's xmin element
            ymin = int(obj.find('bndbox').find('ymin').text)  # The object's ymin element
            xmax = int(obj.find('bndbox').find('xmax').text)  # The object's xmax element
            ymax = int(obj.find('bndbox').find('ymax').text)  # The object's ymax element

            if clazz in classes_filter:
                xml_list.append([filename, image_width, image_height, clazz, xmin, ymin, xmax, ymax])
                classes_names.append(clazz)
    column_names = ['filename', 'image_width', 'image_height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)  # The pandas DataFrame of all the labels
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names

def create_pbtxt(classes):
    """
    Creates a pbtxt file from the given classes.

    :param classes: A list containing the name of the classes
    :return: The pbtxt string, a label to id dict
    """
    pbtxt_content = ''  # The content of the pbtxt file
    label_to_id = {}  # The label name to id dictionary
    for i, class_name in enumerate(classes):
        pbtxt_content = (pbtxt_content
                         + "item {{\n    id: {0}\n    name: '{1}'\n    display_name: '{1}'\n }}\n\n".format(i + 1, class_name))
        label_to_id[class_name] = i

    return pbtxt_content.strip(), label_to_id

def create_tf_example(group, image_dir, label_to_id):
    """
    Create a tf_example for generating a record file from the given data point.

    :param group: The given data point
    :param image_dir: The path to the directory of the images
    :param label_to_id: A dict converting label string to id
    :return: The tf_example
    """
    with tf.io.gfile.GFile(os.path.join(image_dir, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)  # The encoded image to bytes
    image = Image.open(encoded_jpg_io)
    width, height = image.size  # The image's actual size and width

    filename = group.filename.encode('utf8')  # The filename label
    image_format = b'jpg'  # The image format in bytes
    xmins = []  # The image's xmins
    xmaxs = []  # The image's xmaxs
    ymins = []  # The image's ymins
    ymaxs = []  # The image's ymaxs
    classes_text = []  # The name of the classes
    classes = []  # The id of the classes

    # Add each label to the lists
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_to_id[row['class']])

    # Create the tf_example of the file
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def combine_functions(*funcs):
    """
    Combine multiple functions into one.

    :param funcs: The functions to combine
    :return: A combined function
    """
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return combined_func
