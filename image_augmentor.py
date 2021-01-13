import general
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil

#  setting
in_dir = 'train'
out_dir = 'train-aug'
rand_pic_per_raw = 2
datagen = ImageDataGenerator(
    rotation_range=15,  # 0-180
    width_shift_range=0.05,  # 0.-1.
    height_shift_range=0.05,
    shear_range=0.05,  # 0.-1.
    zoom_range=0.05,  # 0.-1.
    horizontal_flip=True,  # True/False
    vertical_flip=False,
    fill_mode='constant',
    cval=255,
)

imgnames = general.get_filenames(in_dir)
k = 0
for imgname in imgnames:
    imgpath = in_dir + '/' + imgname
    img = load_img(imgpath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    basic_name = general.get_basic_filename(imgname)

    shutil.copy(imgpath, out_dir + '/' + basic_name + '_.png')

    for i in range(rand_pic_per_raw):
        prefix = basic_name + '_' + str(i + 1).zfill(3)
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=out_dir,
                                  save_prefix=prefix,
                                  save_format='png'):
            break
    k += 1
    print('done', k)

fns = general.get_filenames(out_dir)

old_name = ''
count = 0
for f in fns:
    flower_name = f.split('(', 1)[0]
    if old_name != flower_name:
        count = 1
        old_name = flower_name

    new_name = flower_name + '(' + str(count) + ').png'
    count += 1
    os.rename(out_dir + '/' + f, out_dir + '/' + new_name)

print('all done')
