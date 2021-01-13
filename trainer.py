import math
import numpy as np
import tensorflow as tf
import h5dataset as h5ds


np.random.seed(0)
tf.set_random_seed(0)

def add_conv(inputs, m_filters, filter_size, name):
    A = tf.layers.conv2d(inputs, m_filters, filter_size, name=name, activation=tf.nn.relu,
                         padding='SAME')
    return A
    
def add_conv_p(inputs, m_filters, filter_size, name):
    A = tf.layers.conv2d(inputs, m_filters, filter_size, name=name, activation=tf.nn.relu,
                         padding='SAME')
    A_p = add_pool(A, name+'_P')
    return A_p
    
def add_conv_bn(inputs, m_filters, filter_size, name, is_training):
    A = tf.layers.conv2d(inputs, m_filters, filter_size, name=name, activation=None,
                         padding='SAME')
    A_bn = tf.layers.batch_normalization(A, name=name+'_BN', training=is_training)
    A_bn_relu = tf.nn.relu(A_bn)
    return A_bn_relu
    
def add_conv_bn_p(inputs, m_filters, filter_size, name, is_training):
    A = tf.layers.conv2d(inputs, m_filters, filter_size, name=name, activation=None,
                         padding='SAME')
    A_bn = tf.layers.batch_normalization(A, name=name+'_BN', training=is_training)
    A_bn_relu = tf.nn.relu(A_bn)
    A_bn_relu_p = add_pool(A_bn_relu, name+'_P')
    return A_bn_relu_p
    
def add_pool(inputs, name):
    A = tf.layers.max_pooling2d(inputs, 2, 2, name=name)
    return A
    
def add_fc(inputs, m_out, name, activation=tf.nn.relu):
    A = tf.layers.dense(inputs, m_out, name=name, activation=activation)
    return A
def lr_decay(lr, i, max_lr, min_lr, decay_speed):
    r = min_lr + (max_lr - min_lr) * math.exp(-i / decay_speed)
    return r
    
learning_rate = 0.0003
pkeep = 0.75
loop = 3000
train_interval = 1
save_path = './save.ckpt'
restore_train_step = False
save_train_step = False

train_ds = h5ds.H5Dataset()
test_ds = h5ds.H5Dataset()
train_ds.load('./train.h5', mini_batch_size=32)
test_ds.load('./test.h5', mini_batch_size=32)
image_size = train_ds.imgsize
flower_type = train_ds.category

with tf.name_scope("Input"):
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name="X")
    X2 = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name="X2")
    X3 = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name="X3")
    Y = tf.placeholder(tf.int32, [None, flower_type], name="Y")
    lr = tf.placeholder(tf.float32, name="learnning_rate")
    pk = tf.placeholder(tf.float32, name="pkeep")
    is_training = tf.placeholder(tf.bool, name="is_training")
    
c1a = add_conv_bn_p( X, 32, 3, "c1a", is_training)
c1b = add_conv_bn_p( X2, 32, 3, "c1b", is_training)
c1c = add_conv_bn_p( X3, 32, 3, "c1c", is_training)
c1 = tf.concat([c1a, c1b, c1c], 3)
c2 = add_conv_bn_p(c1, 64, 3, "c2", is_training)
c3 = add_conv_bn_p(c2, 64, 3, "c3", is_training)
c4 = add_conv_bn_p(c3, 64, 3, "c4", is_training)
c5 = add_conv_bn_p(c4, 64, 3, "c5", is_training)
c6 = add_conv_bn_p(c5, 64, 3, "c6", is_training)
dd = tf.contrib.layers.flatten(c6)
fc1 = add_fc(dd,  32, "fc1")
fc2 = add_fc(fc1, 32, "fc2")
dp = tf.layers.dropout(fc2, rate=pk, training=is_training)
ZL = tf.layers.dense(dp, flower_type)
AL = tf.nn.softmax(ZL)

with tf.name_scope("Loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y)
    cross_entropy = tf.reduce_mean(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(AL, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(extra_update_ops):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
sess = tf.Session()
merger = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log", sess.graph)
saver = tf.train.Saver(tf.global_variables())
m_epoch = 0
max_acc = 0

if restore_train_step:
    saver.restore(sess, save_path)
else:
    sess.run(tf.global_variables_initializer())
    
for i in range(loop):
    batch_train_X, batch_train_Y, is_epoch_start = train_ds.get_multi_batch()
    sess.run(train_step, feed_dict={X: batch_train_X[0], X2: batch_train_X[1], X3: batch_train_X[2],
                                    Y: batch_train_Y, lr: learning_rate, is_training: True, pk: pkeep})
    if i % train_interval == 0:
        a1, c1 = sess.run([accuracy, cross_entropy],
                          feed_dict={X: batch_train_X[0], X2: batch_train_X[1], X3: batch_train_X[2],
                                     Y: batch_train_Y, is_training: False, pk: pkeep}) #
    if is_epoch_start == 1:
        batch_test_X, batch_test_Y, _ = test_ds.get_multi_batch()
        a2, c2 = sess.run([accuracy, cross_entropy],
                          feed_dict={X: batch_test_X[0], X2: batch_test_X[1], X3: batch_test_X[2],
                                     Y: batch_test_Y, is_training: False, pk: pkeep}) #
        max_acc = max(max_acc, a2)
        n_test_correct = 0
        n_test = 0
        while True:
            batch_test_X, batch_test_Y, is_test_epoch_start = test_ds.get_batch(onehot=True)
            batch_test_X = batch_test_X / 255
            a2, c2 = sess.run([accuracy, cross_entropy],
                              feed_dict={X: batch_test_X, Y: batch_test_Y, is_training: False, pk: pkeep})
            n_test += batch_test_X.shape[0]
            n_test_correct += round(batch_test_X.shape[0] * a2)
            if is_test_epoch_start:
                break
        a2 = n_test_correct / n_test
        max_acc = max(max_acc, a2)
        m_epoch += 1
        result = sess.run(merger, feed_dict={X: batch_test_X, Y: batch_test_Y, is_training: False, pk: pkeep}) 
        writer.add_summary(result, i)
        
if save_train_step:
    saver.save(sess, save_path)
    
train_ds.close()
test_ds.close()