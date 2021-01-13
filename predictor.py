import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

def prepare(img):
    t = type(img[0, 0, 0])
    if t == np.float32 or t == np.float64 or t == np.float:
        img = np.floor(img * 255).astype(np.uint8)
    img = np.array([img])
    return img
    
def merge(p1, p2, p3):
    return np.array([p1, p2, p3])
    
def predict(save_dir, meta_name, x, multi):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(save_dir + meta_name)    
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        X =  graph.get_tensor_by_name('X:0')
        X2 = graph.get_tensor_by_name('X2:0')
        X3 = graph.get_tensor_by_name('X3:0')
        AL = graph.get_tensor_by_name('AL:0')
        dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        is_training = graph.get_tensor_by_name('is_training:0')
        predict_label = tf.argmax(AL, 1)
        if multi:
            feed_dict={X: x[0], X2: x[1], X3: x[2], is_training: False, dropout_rate: 1}
        else:
            feed_dict={X: x, X2: x, X3: x, is_training: False, dropout_rate: 1}
        lbl = sess.run(predict_label, feed_dict=feed_dict)
        return lbl