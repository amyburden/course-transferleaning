import os
import glob
import numpy as np

def data_list():
    base = './dataset/256_ObjectCategories/*'
    subfolders = glob.glob(base)
    train_data = []
    val_data = []
    test_data = []
    for subfolder in subfolders[: -1]:
        images = np.array(glob.glob(os.path.join(subfolder,'*.jpg')))
        idx = range(len(images))
        train_idx = idx[:]
        np.random.shuffle(idx)
        train_data.append(images[idx[: len(idx)*8/10]].tolist())
        val_data.append(images[idx[len(idx)*8/10: len(idx)*9/10]].tolist())
        test_data.append(images[idx[len(idx)*9/10:]].tolist())
    return train_data, val_data, test_data
