import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score

from imblearn.metrics import geometric_mean_score

import pandas as pd
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import time

# from tqdm import tqdm

from dataset import BadmintonSwingDataset, BadmintonSwingDataset_trial
from utils import plot_test_auroc, plot_test_confusion_matrix, get_keyframe_positions, compute_avg_position_error

from blockgcn import BlockGCN_4_per_frame, BlockGCN_6_per_frame
from blockgcn import BlockGCN_6_normal_TCN_per_frame


Hyp = namedtuple('Hyp', [
    'num_classes', 'n_epochs', 'lr', 'batch_size'
])
hyp = Hyp(
    num_classes=4,
    n_epochs=500,
    lr=5e-2,
    batch_size=32
)


def train_one_epoch(
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

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).reshape(n, 3, t, -1, 1)
        logits: Tensor = net(x)
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.ravel()
        loss: Tensor = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()
def validate(
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

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).reshape(n, 3, t, -1, 1)
        logits: Tensor = net(x)
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.ravel()
        loss: Tensor = loss_fn(logits, labels)
        total_loss += loss

        # Calculate accuracy
        pred = F.one_hot(torch.argmax(logits, dim=1), hyp.num_classes).bool()
        labels_onehot = F.one_hot(labels).bool()

        # pad mask
        range_t = torch.arange(0, t, device=device).unsqueeze(0).expand(n, -1)
        videos_len = videos_len.unsqueeze(-1)
        mask = (range_t < videos_len).ravel().unsqueeze(-1)
        # mask: (n*t, 1)

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

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).reshape(n, 3, t, -1, 1)
        logits: Tensor = net(x)
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

        # pose need to change to (n, c, t, v, m=1)
        n, t, _ = poses.shape
        x = poses.view(n, t, -1, 3).permute(0, 3, 1, 2).reshape(n, 3, t, -1, 1)
        logits: Tensor = net(x)
        # logits: (n, t, 4)

        logits = logits.view(-1, logits.shape[-1])  # logits: (n*t, 4)
        true_positions = labels.nonzero()[:, 1].reshape(len(labels), -1)  # (n, 3)

        pred = F.softmax(logits, dim=1)

        y_pred.append(pred)
        y_true_positions.append(true_positions)
        all_v_len.append(videos_len)
    
    y_pred = torch.cat(y_pred, dim=0)
    y_true_positions = torch.cat(y_true_positions, dim=0)
    all_v_len = torch.cat(all_v_len)  # default on CPU
    return y_pred, y_true_positions, all_v_len


def train_network(
    net: nn.Module,
    ce_weight: Tensor,
    train_loader,
    val_loader,
    pick_by_key_f1: bool,
    save_path: Path,
    device
):
    loss_fn = nn.CrossEntropyLoss(weight=ce_weight).to(device)
    optimizer = optim.Adam(net.parameters(), lr=hyp.lr)

    best_value = 0.0 if pick_by_key_f1 else torch.inf

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(
            net=net,
            loss_fn=loss_fn,
            optimizer=optimizer,
            loader=train_loader,
            device=device
        )
        val_loss, f1_score_each = validate(
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
        pass

    def prepare_dataloaders(self, more_label_1=True):
        ds_split_seed = 42

        dataset0 = BadmintonSwingDataset(
            pose_path='all_coordinates_list.csv',
            label_path=f'label_all_keyframe.csv',
            videos_len_path='videos_length.npy'
        )
        train_set0, val_set0, test_set0 = random_split(dataset0, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(ds_split_seed))

        dataset1 = BadmintonSwingDataset(
            pose_path='all_coordinates_list.csv',
            label_path=f'label_all_keyframe_add.csv',
            videos_len_path='videos_length.npy'
        )
        train_set1, val_set1, test_set1 = random_split(dataset1, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(ds_split_seed))

        if more_label_1:
            train_set = train_set1
            val_set = val_set1
            test_set = test_set1
        else:
            train_set = train_set0
            val_set = val_set0
            test_set = test_set0

        trial_set = BadmintonSwingDataset_trial(data_trial_dir='data_trial')

        use_cuda = torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=hyp.batch_size,
            shuffle=True,
            pin_memory=use_cuda
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=hyp.batch_size,
            pin_memory=use_cuda
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=hyp.batch_size,
        )

        test_no_add_loader = DataLoader(
            dataset=test_set0,
            batch_size=hyp.batch_size,
        )
        trial_loader = DataLoader(
            dataset=trial_set,
            batch_size=hyp.batch_size,
        )

        self.more_label_1 = more_label_1
        self.model_name_postfix = '' if more_label_1 else '_no_add'

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.test_no_add_loader = test_no_add_loader
        self.trial_loader = trial_loader

        self.use_cuda = use_cuda
        self.device = device

    def get_network_architecture(self, model_name):
        '''
        `model_name`
        - 'blockgcn_4' (868K)
            - using multi-scale TCN
        - 'blockgcn_6' (964K)
            - using multi-scale TCN
        - 'blockgcn_6_normal_tcn' (815K)
        '''
        match model_name:
            case 'blockgcn_4':
                net = BlockGCN_4_per_frame(
                    num_class=hyp.num_classes,
                    num_person=1,
                    in_channels=3,
                    graph_args={
                        'layout': 'mediapipe'
                        # There is no difference between setting
                        # labeling_mode to 'uniform' and to 'spatial' here
                        # because BlockGCN uses only the hop distance from the graph
                        # to build the Static Topological Embedding matrix B.
                    },
                    g_kernel_size=3,
                    n_heads=2,  # default 8 in BlockGCN
                    # If 'n_heads' is bigger, learnable A becomes bigger, but W becomes smaller.
                    # Choosing a suitable value depends on the min hidden channel size.
                    t_kernel_size=9,
                    data_bn=False,
                    last_drop_out=0
                )

            case 'blockgcn_6':
                net = BlockGCN_6_per_frame(
                    num_class=hyp.num_classes,
                    num_person=1,
                    in_channels=3,
                    graph_args={
                        'layout': 'mediapipe'
                        # There is no difference between setting
                        # labeling_mode to 'uniform' and to 'spatial' here
                        # because BlockGCN uses only the hop distance from the graph
                        # to build the Static Topological Embedding matrix B.
                    },
                    g_kernel_size=3,
                    n_heads=8,  # default 8 in BlockGCN
                    # If 'n_heads' is bigger, learnable A becomes bigger, but W becomes smaller.
                    # Choosing a suitable value depends on the min hidden channel size.
                    t_kernel_size=9,
                    data_bn=False,
                    last_drop_out=0
                )

            case 'blockgcn_6_normal_tcn':
                # The hidden channel size here is half of that in BlockGCN default settings
                # because MultiScale_TCN is changed to normal TCN causing larger parameter size.
                net = BlockGCN_6_normal_TCN_per_frame(
                    num_class=hyp.num_classes,
                    num_person=1,
                    in_channels=3,
                    graph_args={
                        'layout': 'mediapipe'
                        # There is no difference between setting
                        # labeling_mode to 'uniform' and to 'spatial' here
                        # because BlockGCN uses only the hop distance from the graph
                        # to build the Static Topological Embedding matrix B.
                    },
                    g_kernel_size=3,
                    n_heads=4,  # default 8 in BlockGCN
                    # If 'n_heads' is bigger, learnable A becomes bigger, but W becomes smaller.
                    # Choosing a suitable value depends on the min hidden channel size.
                    t_kernel_size=9,
                    data_bn=False,
                    last_drop_out=0
                )
            
            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)
    
    def seek_network_weights(self, serial_no=1, pick_by_f1_score=True):
        serial_str = f'_{serial_no}' if serial_no != 1 else ''

        self.model_name_postfix = ('_pick_by_key_f1' if pick_by_f1_score else '') \
                                  + self.model_name_postfix \
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

            self.net = train_network(
                net=self.net,
                ce_weight=ce_weight,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                pick_by_key_f1=pick_by_f1_score,
                save_path=weight_path,
                device=self.device
            )

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

    def test_APE_trial(self):
        y_pred, _, _ = predict_remain_padding(self.net, self.trial_loader, self.device)
        ds: BadmintonSwingDataset_trial = self.trial_loader.dataset
        y_pred_np = y_pred.view(*ds.poses.shape[:2], hyp.num_classes).cpu().numpy()

        y_pos = get_keyframe_positions(y_pred_np, ds.videos_len, no_ordering=True)
        y_pos_ordered = get_keyframe_positions(y_pred_np, ds.videos_len, no_ordering=False)

        ape_score = compute_avg_position_error(y_pos, ds.true_positions, self.save_name_upper)
        ape_score_ordered = compute_avg_position_error(y_pos_ordered, ds.true_positions, self.save_name_upper)
        
        print(f'Test (num_videos: {len(y_pred_np)}) =>')
        pd.set_option('display.float_format', '{:.1f}'.format)
        print(ape_score)
        print(ape_score_ordered)
        print()


if __name__ == '__main__':
    task = Task()
    task.prepare_dataloaders(more_label_1=True)
    task.get_network_architecture(model_name='blockgcn_6')
    task.seek_network_weights(serial_no=2, pick_by_f1_score=True)
    task.test(compute_plot=False, save_plot=False)
    task.test_APE()
    task.test_APE_trial()
