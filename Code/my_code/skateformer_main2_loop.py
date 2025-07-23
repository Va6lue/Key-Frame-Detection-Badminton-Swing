import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score

from imblearn.metrics import geometric_mean_score

from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import time
from datetime import timedelta
from statistics import mean, stdev

# from tqdm import tqdm

from dataset import BadmintonSwingDataset
from utils import plot_test_auroc, plot_test_confusion_matrix, get_keyframe_positions, compute_avg_position_error

from skateformer import SkateFormer_per_frame


Hyp = namedtuple('Hyp', [
    'num_classes', 'n_epochs', 'lr', 'batch_size',
    'T', 'V', 'N', 'L'
])
hyp = Hyp(
    num_classes=4,
    n_epochs=500,
    lr=5e-2,
    batch_size=8,
    T=400,
    N=20,
    V=25,
    L=5
)


def reset_joints_order_before_partitions(x: Tensor, C=3):
    N, S, VC = x.shape
    head_shoulders_elbows = np.arange(5)
    left_hand = np.array([4, 6, 8, 10, 12]) - 1
    right_hand = left_hand + 1
    left_leg = np.array([14, 16, 18, 20, 22]) - 1
    right_leg = left_leg + 1

    new_idx = torch.from_numpy(np.concatenate(
        [head_shoulders_elbows, left_hand, right_hand, left_leg, right_leg]
    )).to(x.device)
    new_x = x.view(N, S, -1, C)
    new_x = torch.index_select(new_x, dim=2, index=new_idx)
    return new_x.view(N, S, -1)  # (N, S, V'C)


def pad_to_T(x: Tensor, label: Tensor):
    # x: (N, S, d)
    # label: (N, S)
    S = x.shape[1]; T = hyp.T
    assert S <= T, 'T is too short.'
    pad_len = T - S
    new_x = F.pad(x, [0, 0, 0, pad_len])
    new_lb = F.pad(label, [0, pad_len])
    return new_x, new_lb


def train_one_epoch_no_pad_loss(
    net: nn.Module,
    loss_fn,
    optimizer: optim.Optimizer,
    loader,
    device
):
    net.train()
    total_loss = 0.0

    for poses, labels, videos_len in loader:
        poses: Tensor = poses.to(device)  # poses: (n, t, v*c)
        labels: Tensor = labels.to(device)  # labels: (n, t)
        videos_len: Tensor = videos_len.to(device)

        # set the parition profiles and fix T
        poses = reset_joints_order_before_partitions(poses)
        poses, labels = pad_to_T(poses, labels)

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).contiguous().view(n, 3, t, -1, 1)
        logits: Tensor = net(x, torch.arange(1, t+1, device=x.device))
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.ravel()
        
        # pad mask
        range_t = torch.arange(0, t, device=device).unsqueeze(0).expand(n, -1)
        videos_len = videos_len.unsqueeze(-1)
        mask = (range_t < videos_len).ravel().unsqueeze(-1)
        # mask: (n*t, 1)

        pure_loss: Tensor = loss_fn(logits, labels) * mask
        loss = pure_loss.sum() / videos_len.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()
def validate_no_pad_loss(
    net: nn.Module,
    loss_fn,
    loader,
    device
):
    net.eval()
    total_loss = 0.0
    cum_tp = torch.zeros(hyp.num_classes)
    cum_tn = torch.zeros(hyp.num_classes)
    cum_fp = torch.zeros(hyp.num_classes)
    cum_fn = torch.zeros(hyp.num_classes)

    for poses, labels, videos_len in loader:
        poses: Tensor = poses.to(device)  # poses: (n, t, v*c)
        labels: Tensor = labels.to(device)  # labels: (n, t)
        videos_len: Tensor = videos_len.to(device)

        # set the parition profiles and fix T
        poses = reset_joints_order_before_partitions(poses)
        poses, labels = pad_to_T(poses, labels)

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).contiguous().view(n, 3, t, -1, 1)
        logits: Tensor = net(x, torch.arange(1, t+1, device=x.device))
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.ravel()

        # pad mask
        range_t = torch.arange(0, t, device=device).unsqueeze(0).expand(n, -1)
        videos_len = videos_len.unsqueeze(-1)
        mask = (range_t < videos_len).ravel().unsqueeze(-1)
        # mask: (n*t, 1)

        pure_loss: Tensor = loss_fn(logits, labels) * mask
        total_loss += pure_loss.sum() / videos_len.sum()

        # Calculate accuracy
        pred = F.one_hot(torch.argmax(logits, dim=1), hyp.num_classes).bool()
        labels_onehot = F.one_hot(labels).bool()

        tp = torch.sum(pred & labels_onehot & mask, dim=0)
        tn = torch.sum(~pred & ~labels_onehot & mask, dim=0)

        fp = torch.sum(pred & ~labels_onehot & mask, dim=0)
        fn = torch.sum(~pred & labels_onehot & mask, dim=0)

        cum_tp += tp.cpu()
        cum_tn += tn.cpu()
        cum_fp += fp.cpu()
        cum_fn += fn.cpu()

    val_loss = total_loss / len(loader)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0

    return val_loss, f1_score


@torch.no_grad()
def predict(
    net: nn.Module,
    loader,
    device
):
    net.eval()
    y_pred = []
    y_true = []
    for poses, labels, videos_len in loader:
        poses: Tensor = poses.to(device)  # poses: (n, t, v*c)
        labels: Tensor = labels.to(device)  # labels: (n, t)
        videos_len: Tensor = videos_len.to(device)

        # set the parition profiles and fix T
        poses = reset_joints_order_before_partitions(poses)
        poses, labels = pad_to_T(poses, labels)

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).contiguous().view(n, 3, t, -1, 1)
        logits: Tensor = net(x, torch.arange(1, t+1, device=x.device))
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])  # logits: (n*t, 4)
        labels = labels.ravel()  # labels: (n*t,)

        pred = F.softmax(logits, dim=1)

        # pad mask
        range_t = torch.arange(0, t, device=device).unsqueeze(0).expand(n, -1)
        videos_len = videos_len.unsqueeze(-1)
        mask = (range_t < videos_len).ravel()

        y_pred.append(pred[mask])
        y_true.append(labels[mask])
    
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    return y_pred, y_true


@torch.no_grad()
def predict_remain_padding(
    net: nn.Module,
    loader,
    device
):
    '''
    Return:
        `y_pred` shape: (N*T, 4)
        `y_true_positions` shape: (N, 3)
    '''
    net.eval()
    y_pred = []
    y_true_positions = []
    all_v_len = []
    for poses, labels, videos_len in loader:
        poses: Tensor = poses.to(device)  # poses: (n, t, v*c)
        labels: Tensor = labels.to(device)  # labels: (n, t)
        videos_len: Tensor

        # set the parition profiles and fix T
        poses = reset_joints_order_before_partitions(poses)
        poses, labels = pad_to_T(poses, labels)

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).contiguous().view(n, 3, t, -1, 1)
        logits: Tensor = net(x, torch.arange(1, t+1, device=x.device))
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])  # logits: (n*t, 4)
        true_positions = labels.nonzero()[:, 1].contiguous().view(len(labels), -1)  # (n, 3)

        pred = F.softmax(logits, dim=1)

        y_pred.append(pred)
        y_true_positions.append(true_positions)
        all_v_len.append(videos_len)
    
    y_pred = torch.cat(y_pred, dim=0)
    y_true_positions = torch.cat(y_true_positions, dim=0)
    all_v_len = torch.cat(all_v_len)  # default on CPU
    return y_pred, y_true_positions, all_v_len


def train_network(
    net: SkateFormer_per_frame,
    ce_weight: Tensor,
    train_loader,
    val_loader,
    pick_by_key_f1: bool,
    save_path: Path,
    device
):
    loss_fn = nn.CrossEntropyLoss(weight=ce_weight, reduction='none').to(device)

    para_group_dic = [
        {
            'params': [v for k, v in net.named_parameters() if k not in net.no_weight_decay()],
            'weight_decay': 0.0  # original: 5e-4
        },
        {
            'params': [v for k, v in net.named_parameters() if k in net.no_weight_decay()],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(para_group_dic, lr=hyp.lr)

    best_value = 0.0 if pick_by_key_f1 else torch.inf

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch_no_pad_loss(
            net=net,
            loss_fn=loss_fn,
            optimizer=optimizer,
            loader=train_loader,
            device=device
        )
        val_loss, f1_score_each = validate_no_pad_loss(
            net=net,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device
        )
        t1 = time.time()

        f1_score_ls = [f'{e:.3f}' for e in f1_score_each]
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '\
              f'val_loss={val_loss:.3f}, f1_score={f1_score_ls} '\
              f'- {t1 - t0:.2f} s', flush=True)

        if pick_by_key_f1:
            key_f1 = f1_score_each[1:].mean()
            to_pick = best_value < key_f1
            cur_value = key_f1
        else:
            to_pick = best_value > val_loss
            cur_value = val_loss

        # Pick the best model parameters
        if to_pick:
            best_value = cur_value
            best_state = deepcopy(net.state_dict())
            print(f'Picked! => Best value {cur_value:.3f}')

    torch.save(best_state, str(save_path))
    net.load_state_dict(best_state)
    return net


class Task:
    def __init__(self) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

    def cross_validate(self, k_splits=5, n_times=3):
        all_f1_mu = []
        all_f1_std = []
        all_APE_mu = []
        all_APE_std = []

        random_states = list(range(n_times))
        for r_state in random_states:
            macro_f1_scores = []
            avg_APE_scores = []
            for k in range(k_splits):
                self.prepare_dataloaders(k=k, k_splits=k_splits, random_state=r_state)
                self.get_network_architecture(model_name='skateformer')
                self.seek_network_weights(model_info=f'no_pad_loss_s_{r_state}', serial_no=1, pick_by_f1_score=True)
                macro_f1 = self.test(compute_plot=False, save_plot=True)
                avg_APE = self.test_APE()
                
                macro_f1_scores.append(macro_f1)
                avg_APE_scores.append(avg_APE)
            
            mu_macro_f1 = mean(macro_f1_scores)
            mu_avg_APE = mean(avg_APE_scores)
            std_macro_f1 = stdev(macro_f1_scores)
            std_avg_APE = stdev(avg_APE_scores)

            all_f1_mu.append(mu_macro_f1)
            all_f1_std.append(std_macro_f1)
            all_APE_mu.append(mu_avg_APE)
            all_APE_std.append(std_avg_APE)

        print(f'{k_splits}-fold x {n_times} times cross validation =>')
        print('<F1-score>')
        print('Mean:', [f'{e:.3f}' for e in all_f1_mu])
        print('Std:', [f'{e:.3f}' for e in all_f1_std])
        print('<APE>')
        print('Mean:', [f'{e:.2f}' for e in all_APE_mu])
        print('Std:', [f'{e:.2f}' for e in all_APE_std])
        return mean(all_f1_mu), mean(all_f1_std), mean(all_APE_mu), mean(all_APE_std)

    def prepare_dataloaders(
        self,
        k,
        k_splits=5,
        coordinate_dim=3,
        more_label_1=True,
        random_state=None
    ):
        match coordinate_dim:
            case 3:
                pose_path = 'all_coordinates_list.csv'
            case 2:
                pose_path = 'all_coordinates_xy.csv'  # The z values are set to 0 in this .csv file.
            case _:
                raise ValueError

        kf = KFold(n_splits=k_splits, shuffle=(random_state is not None), random_state=random_state)

        dataset0 = BadmintonSwingDataset(
            pose_path=pose_path,
            label_path=f'label_all_keyframe.csv',
            videos_len_path='videos_length.npy',
            cleaned=True
        )
        train_index, test_index = list(kf.split(dataset0))[k]
        half = len(test_index) // 2
        val_index = test_index[:half]
        test_index = test_index[half:]

        train_set0 = Subset(dataset0, train_index)
        val_set0 = Subset(dataset0, val_index)
        test_set0 = Subset(dataset0, test_index)

        dataset1 = BadmintonSwingDataset(
            pose_path=pose_path,
            label_path=f'label_all_keyframe_add.csv',
            videos_len_path='videos_length.npy',
            cleaned=True
        )
        train_set1 = Subset(dataset1, train_index)
        val_set1 = Subset(dataset1, val_index)
        test_set1 = Subset(dataset1, test_index)

        if more_label_1:
            train_set = train_set1
            val_set = val_set1
            test_set = test_set1
        else:
            train_set = train_set0
            val_set = val_set0
            test_set = test_set0

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=hyp.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=hyp.batch_size,
            pin_memory=self.use_cuda
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=hyp.batch_size,
        )

        test_no_add_loader = DataLoader(
            dataset=test_set0,
            batch_size=hyp.batch_size,
        )

        self.k = k
        self.more_label_1 = more_label_1
        self.model_name_postfix = '' if more_label_1 else '_no_add'
        if coordinate_dim == 2:
            self.model_name_postfix += '_2d'

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.test_no_add_loader = test_no_add_loader

    def get_network_architecture(self, model_name):
        '''
        `model_name`
        - 'skateformer-tiny' (538K)
        - 'skateformer' (766K)
        '''
        T = hyp.T; V = hyp.V  # (T frames, V joints) in total
        N = hyp.N; L = hyp.L  # (N frames, L joints) in first partition

        match model_name:
            case "skateformer-tiny":
                net = SkateFormer_per_frame(
                    in_channels=3,
                    channels=(64, 128),
                    depths=(2, 2),
                    num_classes=4,
                    num_frames=T,
                    num_points=V,
                    num_people=1,
                    num_heads=8,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True
                )

            case 'skateformer':
                net = SkateFormer_per_frame(
                    in_channels=3,
                    channels=(64, 128),
                    depths=(2, 3),
                    num_classes=4,
                    num_frames=T,
                    num_points=V,
                    num_people=1,
                    num_heads=32,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True
                )

            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)
    
    def seek_network_weights(self, model_info='', serial_no=1, pick_by_f1_score=True):
        serial_str = f'_{serial_no}' if serial_no != 1 else ''
        model_info = f'_{model_info}' if model_info != '' else ''

        self.model_name_postfix = ('_pick_by_key_f1' if pick_by_f1_score else '') \
                                  + self.model_name_postfix \
                                  + model_info \
                                  + f'_k_{self.k}'\
                                  + serial_str
        self.save_name = self.model_name + self.model_name_postfix
        self.save_name_upper = self.model_name.upper() + self.model_name_postfix

        weight_path = Path(f'my_code/weights/{self.save_name}.pt')
        if weight_path.exists():
            self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))
        else:
            if self.more_label_1:
                ce_weight = torch.tensor([0.2, 1.0, 3.0, 3.0])
            else:
                ce_weight = torch.tensor([0.2, 3.0, 3.0, 3.0])

            train_t0 = time.time()
            self.net = train_network(
                net=self.net,
                ce_weight=ce_weight,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                pick_by_key_f1=pick_by_f1_score,
                save_path=weight_path,
                device=self.device
            )
            train_t1 = time.time()
            t = timedelta(seconds=int(train_t1 - train_t0))
            print(f'Total training time: {t}')

    def test(self, compute_plot=True, save_plot=True):
        y_pred, y_true = predict(self.net, self.test_loader, self.device)

        # Plot AUROC and Confusion matrix
        y_pred_np = y_pred.cpu().numpy()
        y_true_one_hot_np = F.one_hot(y_true).cpu().numpy()
        if compute_plot:
            plot_test_auroc(
                y_true_one_hot_np, y_pred_np, self.model_name,
                save_name=self.save_name, save=save_plot
            )
            plot_test_confusion_matrix(
                y_true_one_hot_np, y_pred_np, self.model_name,
                save_name=self.save_name, save=save_plot
            )

        # Compute each class F1-score
        f1_score_each = multiclass_f1_score(y_pred, y_true, num_classes=hyp.num_classes, average=None)
        f1_score_all = torch.cat([f1_score_each, f1_score_each.mean(dim=0, keepdim=True)]).unsqueeze(0).cpu().numpy()
        df = pd.DataFrame(
            data=f1_score_all,
            columns=['keyframe0', 'keyframe1', 'keyframe2', 'keyframe3', 'avg'],
        )
        df[self.save_name_upper] = 'F1-score'
        df.set_index(self.save_name_upper, drop=True, inplace=True)
        
        print(f'Test (num_frames: {len(y_pred)}) =>')
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df)
        print()

        # Compute multiclass G-mean
        y_pred_class_np = torch.argmax(y_pred, dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        g_mean_multiclass = geometric_mean_score(y_true_np, y_pred_class_np, average='multiclass')
        print('G-mean (multiclass):', f'{g_mean_multiclass:.3f}')
        print()

        return f1_score_each[1:].mean().item()

    def test_APE(self):
        y_pred, y_true_positions, videos_len = predict_remain_padding(self.net, self.test_no_add_loader, self.device)

        y_pred_np = y_pred.view(len(y_true_positions), -1, hyp.num_classes).cpu().numpy()
        y_true_positions_np = y_true_positions.cpu().numpy()
        videos_len_np = videos_len.numpy()  # This is already on CPU.

        y_pos = get_keyframe_positions(y_pred_np, videos_len_np, no_ordering=True)
        y_pos_ordered = get_keyframe_positions(y_pred_np, videos_len_np, no_ordering=False)

        ape_score = compute_avg_position_error(y_pos, y_true_positions_np, self.save_name_upper)
        ape_score_ordered = compute_avg_position_error(y_pos_ordered, y_true_positions_np, self.save_name_upper)
        
        print(f'Test (num_videos: {len(y_pred_np)}) =>')
        pd.set_option('display.float_format', '{:.1f}'.format)
        print(ape_score)
        print(ape_score_ordered)
        print()

        return ape_score_ordered['avg'].values[0]


if __name__ == '__main__':
    task = Task()
    mean_f1_mu, mean_f1_std, mean_APE_mu, mean_APE_std = \
        task.cross_validate(k_splits=5, n_times=3)
    
    print('Summary:')
    print(f'Mean F1-score: {mean_f1_mu:.3f} ± {mean_f1_std:.3f}')
    print(f'Mean APE: {mean_APE_mu:.2f} ± {mean_APE_std:.2f}')
