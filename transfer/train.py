import csv
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


random.seed(0)
np.random.seed(0)

N_FEATURE = 4096
labels = []
codes = None

with open('extract_all/codes_orchid') as f:
    codes = np.fromfile(f, dtype=np.float32).reshape((-1, N_FEATURE))

with open('extract_all/labels_orchid') as f:
    reader = csv.reader(f, delimiter='\n')
    for row in reader:
        labels.append(row[0])

        
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
    print(buf.shape)
    multi_codes = buf if multi_codes is None else np.concatenate((multi_codes, buf), axis=0)
    
# codes = multi_codes  # switch

# convert to one hot
lb = LabelBinarizer()
lb.fit(labels)

# split to train & validation
labels_vecs = lb.transform(labels)
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
train_idx, val_idx = next(ss.split(codes, labels))


# split validation to validation & test
half_val_len = int(len(val_idx))
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

print("Train shapes (x, y):", train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

inputs_ = tf.placeholder(tf.float32, shape=[None, N_FEATURE])
inputs_2 = tf.placeholder(tf.float32, shape=[None, N_FEATURE])
inputs_3 = tf.placeholder(tf.float32, shape=[None, N_FEATURE])

labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

fc1 = tf.contrib.layers.fully_connected(inputs_, 288)
fc2 = tf.contrib.layers.fully_connected(fc1, 128)

logits = tf.contrib.layers.fully_connected(fc2, labels_vecs.shape[1], activation_fn=None)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(cost)

predicted = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, n_batches=10):
    batch_size = len(x)//n_batches
    
    for ii in range(0, n_batches*batch_size, batch_size):
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y
        
        
epochs = 50
iteration = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for x, y in get_batches(train_x, train_y):
            feed = {inputs_: x,
                    labels_: y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e+1, epochs),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1
            
            if iteration % 5 == 0:
                feed = {inputs_: val_x,
                        labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed)
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
                      
    saver.save(sess, "checkpoints/save.ckpt")
        
        
