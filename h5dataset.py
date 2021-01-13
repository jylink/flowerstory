import h5py
import numpy as np
import random
import matplotlib.image as mpimg

class H5Dataset:
    def __init__(self):
        self.dataset = None
        self.size = 0
        self.category = 0
        self.imgsize = 0
        self.mini_batches = []
        self.mini_batch_size = 0
        
    def load(self, src, mini_batch_size=1):
        self.dataset = h5py.File(src, 'r')
        self.size = self.dataset.attrs['size']
        self.category = self.dataset.attrs['category']
        self.imgsize = self.dataset.attrs['imgsize']
        self.mini_batch_size = mini_batch_size
        assert self.size >= mini_batch_size
        
    def close(self):
        self.dataset.close()
        self.__init__()
        
    def get_batch(self, onehot=True):
        assert self.dataset is not None
        batch_remain = len(self.mini_batches)
        if batch_remain == 0:
            self.mini_batches = H5Dataset.__random_mini_batches(self.size, self.mini_batch_size)
        mini_batch = self.mini_batches[0]
        del self.mini_batches[0]
        mini_batch_x = []
        mini_batch_y = []
        for i in mini_batch:
            x = self.dataset['images'][i]
            y = self.dataset['labels'][i]
            mini_batch_x.append(x)
            mini_batch_y.append(y)
        if onehot:
            mini_batch_y = H5Dataset.__to_onehot(mini_batch_y, self.category)
            mini_batch_y = np.array(mini_batch_y)
        else:
            mini_batch_y = np.squeeze(np.array(mini_batch_y)).reshape(-1)
        mini_batch_x = np.array(mini_batch_x)
        return mini_batch_x, mini_batch_y, batch_remain
        
    def get_multi_batch(self, onehot=True):
        assert self.dataset is not None
        batch_remain = len(self.mini_batches)
        if batch_remain == 0:
            end_indexs = self.dataset['end_index'][:].reshape(-1)
            self.mini_batches = H5Dataset.__random_multi_batches(self.size, self.mini_batch_size, end_indexs)
        mini_batch = self.mini_batches[0]
        del self.mini_batches[0]
        mini_batch_x = [[], [], []]
        mini_batch_y = [[], [], []]
        for group in mini_batch:
            for i in range(3):
                index = group[i]
                x = self.dataset['images'][index]
                y = self.dataset['labels'][index]
                mini_batch_x[i].append(x)
                mini_batch_y[i].append(y)
        for i in range(3):
            mini_batch_x[i] = np.array(mini_batch_x[i])
        mini_batch_y = mini_batch_y[0]
        if onehot:
            mini_batch_y = H5Dataset.__to_onehot(mini_batch_y, self.category)
            mini_batch_y = np.array(mini_batch_y)
        else:
            mini_batch_y = np.squeeze(np.array(mini_batch_y)).reshape(-1)
        mini_batch_x = np.array(mini_batch_x)
        return mini_batch_x, mini_batch_y, batch_remain
        
    def show(self):
        print('size:', self.size)
        print('category:', self.category)
        print('imgsize:', self.imgsize)
        print('end index:', self.dataset['end_index'][:])
        
    @staticmethod
    def build(src, dst):
        dataset = None
        labels = None
        images = None
        end_indexs = None
        cnt = 0
        max_label = 0
        imgsize = 0
        sum_of_mean = None
        sum_of_std = None
        with open(src, 'r') as f:
            for line in f:
                tokens = line.strip('\n').split(' ', 1)
                img = mpimg.imread(tokens[1])
                lbl = int(tokens[0])
                img = H5Dataset.__to_uint8(img[:, :, :3])
                if not dataset:
                    imgsize = img.shape[0]
                    sum_of_mean = np.zeros((imgsize, imgsize, 3), dtype=np.float32)
                    sum_of_std = np.zeros((imgsize, imgsize, 3), dtype=np.float32)
                    dataset, images, labels, end_indexs = H5Dataset.__create_h5(dst, imgsize)
                assert img.shape == (imgsize, imgsize, 3)
                images.resize((cnt + 1, imgsize, imgsize, 3))
                labels.resize((cnt + 1, 1))
                images[cnt] = img
                labels[cnt] = lbl
                if max_label < lbl:
                    end_indexs.resize((max_label + 1, 1))
                    end_indexs[max_label] = cnt
                    max_label = lbl
                sum_of_mean += img
                sum_of_std += img * img
                cnt += 1
                
        if dataset is not None:
            dataset.attrs['size'] = cnt
            dataset.attrs['category'] = max_label + 1
            dataset.attrs['imgsize'] = imgsize
            end_indexs.resize((max_label + 1, 1))
            end_indexs[max_label] = cnt
            dataset['mean'][:] = sum_of_mean / cnt
            dataset['std'][:] = sum_of_std / cnt
            dataset.close()
            
    @staticmethod
    def __random_mini_batches(size, batch_size):
        batches = []
        permutation = np.random.permutation(size)
        n_complete_batches = size // batch_size
        for k in range(n_complete_batches):
            l = batch_size * k
            r = batch_size * (k + 1)
            batch = permutation[l:r]
            batches.append(batch)
        if size % batch_size != 0:
            l = batch_size * n_complete_batches
            r = size
            batch = permutation[l:r]
            batches.append(batch)
        return batches
        
    @staticmethod
    def __random_multi_batches(size, batch_size, end_indexs):
        NUM_PER_GROUP = 3
        assert size % NUM_PER_GROUP == 0
        size = size // NUM_PER_GROUP
        batches = []
        permutation = []
        last_index = 0
        
        for index in end_indexs:
            batch = np.random.permutation(index - last_index) + last_index
            last_index = index
            batch = batch.reshape((-1, NUM_PER_GROUP))
            for k in range(batch.shape[0]):
                permutation.append(batch[k])
        random.shuffle(permutation)
        permutation = np.array(permutation)
        n_complete_batches = size // batch_size
        
        for k in range(n_complete_batches):
            l = batch_size * k
            r = batch_size * (k + 1)
            batch = permutation[l:r]
            batches.append(batch)
        if size % batch_size != 0:
            l = batch_size * n_complete_batches
            r = size
            batch = permutation[l:r]
            batches.append(batch)
        return batches
        
    @staticmethod
    def __to_onehot(labels, n_category):
        onehots = []
        for lbl in labels:
            categ = lbl[0]
            oneh = np.zeros(n_category)
            oneh[categ] = 1
            onehots.append(oneh)
        return onehots
        
    @staticmethod
    def __to_uint8(img):
        t = type(img[0, 0, 0])
        if t == np.float32 or t == np.float64 or t == np.float:
            img = np.floor(img * 255).astype(np.uint8)
        return img
        
    @staticmethod
    def __create_h5(dst, imgsize):
        dataset = h5py.File(dst, 'w')
        images = dataset.create_dataset('images', dtype=np.uint8, shape=(1, imgsize, imgsize, 3),
                                        maxshape=(None, imgsize, imgsize, 3))
        labels = dataset.create_dataset('labels', dtype=np.int8, shape=(1, 1), maxshape=(None, 1))
        end_indexs = dataset.create_dataset('end_index', dtype=np.int32, shape=(1, 1), maxshape=(None, 1))
        mean = dataset.create_dataset('mean', dtype=np.float32, shape=(imgsize, imgsize, 3))
        std = dataset.create_dataset('std', dtype=np.float32, shape=(imgsize, imgsize, 3))
        return dataset, images, labels, end_indexs