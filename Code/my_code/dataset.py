from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import BorderlineSMOTE

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(x_path, y_path, plot_class_distribution=False):
    ds_x = pd.read_csv(x_path)
    ds_y = pd.read_csv(y_path)

    # Preprocess data_x
    data_x = ds_x['coordinate'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
    data_x = np.stack(data_x)

    data_y = LabelEncoder().fit_transform(ds_y['keyframe'])

    print('Class Distribution =>')
    counter = Counter(data_y)
    for k, v in counter.items():
        per = v / len(data_y) * 100
        print('class=%d, n=%d (%.3f%%)' % (k, v, per))
    print()

    if plot_class_distribution:
        plt.bar(counter.keys(), counter.values())
        plt.show()

    return data_x, data_y


def load_data_trial(data_trial_dir: Path):
    '''Load data_trial for testing APE.
    
    Return:
        pad_videos: (N, T, d)
        videos_len: (N)
        true_positions: (N, 3)
    '''
    # Load ground truth
    true_pos_path = data_trial_dir/'true_positions.npy'
    true_positions = np.load(str(true_pos_path))

    # Load data
    data_trial_dir /= 'Coordinate'
    
    videos = []
    videos_len = []

    for i in range(1, 11):
        path = data_trial_dir/f'all_coordinates_{i:02}.csv'
        data = pd.read_csv(str(path))
        data = data['coordinate'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
        data = np.stack(data)
        videos.append(data)
        videos_len.append(len(data))

    videos_len = np.stack(videos_len)
    max_len = videos_len.max()

    pad_videos = []
    for v in videos:
        v: np.ndarray
        pad_len = max_len - len(v)
        pad_videos.append(np.concatenate([
            v, np.zeros((pad_len, v.shape[1]), dtype=v.dtype)
        ], axis=0))
    
    # (N, T, d), (N), (N, 3)
    return np.stack(pad_videos), videos_len, true_positions


def split_data(data_x, data_y, train_ratio=0.8):
    X_train, X_temp, y_train, y_temp = train_test_split(data_x, data_y, test_size=(1-train_ratio), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Check shapes of the split data
    print('After Splitting =>')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


def oversample(X_train, y_train):
    oversampling = BorderlineSMOTE(random_state=42, k_neighbors=5, kind='borderline-1')  # 有borderline-2
    X_train_resampled, y_train_resampled = oversampling.fit_resample(X_train, y_train)

    print('After Oversampling =>')
    print('befor:', Counter(y_train))
    print('after:', Counter(y_train_resampled))
    print()

    return X_train_resampled, y_train_resampled


class BadmintonSwingDataset(Dataset):
    def __init__(self, pose_path, label_path, videos_len_path, cleaned=False) -> None:
        super().__init__()
        data_x, data_y = load_data(pose_path, label_path)
        data_x: np.ndarray; data_y: np.ndarray
        
        videos_len: np.ndarray = np.load(videos_len_path)
        videos_len_cum = np.cumsum(videos_len)
        
        max_len = videos_len.max()
        X = []; y = []
        i = 0
        for v_len, j in zip(videos_len, videos_len_cum):
            pad_len = max_len - v_len
            X.append(np.concatenate([
                data_x[i:j],
                np.zeros((pad_len, data_x.shape[1]), dtype=data_x.dtype)
            ], axis=0))
            y.append(np.concatenate([
                data_y[i:j],
                np.zeros((pad_len), dtype=data_y.dtype)
            ]))
            i = j

        self.poses = np.stack(X)  # pose: (N, T, d)
        self.labels = np.stack(y)  # label: (N, T)
        self.videos_len = videos_len  # (N)

        # delete invalid data
        if cleaned:
            dirt = []
            for i, lb in enumerate(self.labels):
                p1 = len(lb[lb == 1].nonzero()[0])
                p2 = len(lb[lb == 2].nonzero()[0])
                p3 = len(lb[lb == 3].nonzero()[0])
                if p1 < 1 or p2 != 1 or p3 != 1 or i == 30:
                    # i == 30, there is a KF1 in 'add' but there isn't any in 'no_add'.
                    dirt.append(i)

            mask = np.ones_like(videos_len, dtype=bool)
            mask[dirt] = False
            self.poses = self.poses[mask]
            self.labels = self.labels[mask]
            self.videos_len = self.videos_len[mask]

            print('Delete unclean entry count:', len(dirt))
            print('Entry count:', len(self.labels))
            print()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.poses[i], self.labels[i], self.videos_len[i]


class BadmintonSwingDataset_trial(Dataset):
    def __init__(self, data_trial_dir) -> None:
        super().__init__()
        data, videos_len, true_positions = load_data_trial(Path(data_trial_dir))
        
        self.poses = data  # pose: (N, T, d)
        self.videos_len = videos_len

        self.true_positions = true_positions  # (N, 3)

    def __len__(self):
        return len(self.videos_len)
    
    def __getitem__(self, i):
        none = np.zeros(self.poses.shape[1])
        none[:3] = np.arange(3)
        return self.poses[i], none, np.zeros(1)


if __name__ == '__main__':
    dataset = BadmintonSwingDataset(
        pose_path='all_coordinates_list.csv',
        label_path='label_all_keyframe.csv',
        videos_len_path='videos_length.npy',
        cleaned=False
    )
    # dataset = BadmintonSwingDataset_trial('data_trial')

    print(len(dataset[30][1][dataset[30][1] == 1]))
    print(len(dataset))

    unclean = []
    for i, (pose, label, video_len) in enumerate(dataset):
        p1 = len(label[label == 1].nonzero()[0])
        p2 = len(label[label == 2].nonzero()[0])
        p3 = len(label[label == 3].nonzero()[0])
        if p1 < 1 or p2 != 1 or p3 != 1:
            unclean.append(i)
        # if p1 != 1:
        #     print('p1', i, p1)
        # if p2 != 1:
        #     print('p2', i, p2)
        # if p3 != 1:
        #     print('p3', i, p3)
    print('unclean:', unclean)
