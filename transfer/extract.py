import os
import numpy as np
import tensorflow as tf
import csv
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'disease_photos_split/test/orchid/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
batch_size = 10
codes_list = []
labels = []
batch = []
codes = None

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)
    
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            img = utils.load_image(os.path.join(class_path, file))
            img = img[:,:,:3]
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)
                feed_dict = {input_: images}
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))
                batch = []
                print('{} images processed'.format(ii))
                
with open('extract_test/codes_orchid', 'w') as f:
    codes.tofile(f)

with open('extract_test/labels_orchid', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)