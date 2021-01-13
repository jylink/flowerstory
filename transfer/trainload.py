import csv
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

N_FEATURE = 4096

def get_dataset(code_path, label_path, is_multi):
   
    labels = []
    codes = None
    with open(code_path) as f:
        codes = np.fromfile(f, dtype=np.float32).reshape((-1, N_FEATURE))
    with open(label_path) as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            labels.append(row[0])
#     print(codes.shape)
#     print(len(labels))

    # convert to multi
    categories = []
    prev = None
    left = right = 0
    for l in labels:
        if prev is not None and prev != l:
            categories.append([left, right])
            left = right
        prev = l
        right += 1
    categories.append([left, right])

    multi_codes = None
    for ca in categories:
        buf = None
        cd = codes[ca[0]:ca[1]]
        for t in range(3):
            k = np.arange(cd.shape[0])
            random.shuffle(k)
            if buf is None:
                buf = cd[k]
            else:
                buf = np.concatenate((buf, cd[k]), axis=1)
        multi_codes = buf if multi_codes is None else np.concatenate((multi_codes, buf), axis=0)

    if is_multi:
        codes = multi_codes

    # one hot
    lb = LabelBinarizer()
    lb.fit(labels)

    labels_vecs = lb.transform(labels)
    train_idx = np.arange(len(labels)).tolist()
    random.shuffle(train_idx)
    
    train_x, train_y = codes[train_idx], labels_vecs[train_idx]
    print("Train shapes (x, y):", train_x.shape, train_y.shape)
    return train_x, train_y
    
    
    
    