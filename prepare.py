import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from PIL import Image

def read_and_show_image(path):
    img = load_image(path)
    show_image(img)
    
def load_image(path):
    img = mpimg.imread(path)
    return img
    
def show_image(image):
    plt.imshow(image)
    plt.show()
    
def get_filenames(path):
    fn = []
    for (dirpath, dirnames, filenames) in walk(path):
        fn.extend(filenames)
        break
    return fn
    
def get_file_paths(path):
    fp = get_filenames(path)
    for i in range(0, len(fp)):
        fp[i] = path + '/' + fp[i]
    return fp
    
def get_basic_filename(filename):
    base = filename.split('.', 1)
    return base[0]
    
def crop_images(width, height, load_path, save_path, mode):
    image_names = get_filenames(load_path)
    for image_name in image_names:
        img_orig = Image.open(load_path + image_name)
        width_orig, height_orig = img_orig.size
        center_x = width_orig / 2
        center_y = height_orig / 2
        p1_x, p1_y, p2_x, p2_y = 0, 0, 0, 0
        if mode == 'ratio':
            ratio = width / height
            ratio_orig = width_orig / height_orig
            width_target, height_target = 0, 0
            if ratio > ratio_orig:
                width_target = width_orig
                height_target = width_orig / ratio
            else:
                width_target = height_orig * ratio
                height_target = height_orig
            p1_x = center_x - width_target / 2
            p1_y = center_y - height_target / 2
            p2_x = center_x + width_target / 2
            p2_y = center_y + height_target / 2
        elif mode == 'pixel':
            p1_x = center_x - width / 2
            p1_y = center_y - height / 2
            p2_x = center_x + width / 2
            p2_y = center_y + height / 2
        output_path = save_path + image_name
        img = img_orig.crop((p1_x, p1_y, p2_x, p2_y))
        img.save(output_path)
        
def convert_images(load_path, save_path, new_extension, new_name=''):
    image_names = get_filenames(load_path)
    count = 1
    for image_name in image_names:
        input = load_path + image_name
        img = Image.open(input)
        if new_name == '':
            base = get_basic_filename(image_name)
            output = save_path + base + '.' + new_extension
        else:
            output = save_path + new_name + '_' + str(count).zfill(4) + '.' + new_extension
            count += 1
        img.save(output, new_extension)
        
def resize_images(width, height, load_path, save_path):
    image_names = get_filenames(load_path)
    for image_name in image_names:
        input = load_path + image_name
        img = Image.open(input)
        img = img.resize((width, height), Image.ANTIALIAS)
        output = save_path + image_name
        img.save(output)
        
def split_dataset(load_path, save_path, partial, seed=0):
    image_names = get_filenames(load_path)
    np.random.seed(seed)
    image_names = np.array(image_names)
    np.random.shuffle(image_names)
    count = int(image_names.shape[0] * partial)
    for i in range(count):
        input = load_path + image_names[i]
        output = save_path + image_names[i]
        shutil.move(input, output)