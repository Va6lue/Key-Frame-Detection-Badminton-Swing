from sklearn.metrics import roc_curve, auc, confusion_matrix

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Union


def compute_micro_auroc(y_true: np.ndarray, y_pred: np.ndarray):
    '''`y_true` should be one-hot encoded.'''
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)
    return (
        {'micro': fpr},
        {'micro': tpr},
        {'micro': roc_auc}
    )


def compute_macro_auroc(y_true: np.ndarray, y_pred: np.ndarray):
    '''`y_true` should be one-hot encoded.'''
    fpr = dict()  # False Positive Rate
    tpr = dict()  # True Positive Rate
    roc_auc = dict()  # AUC
    num_classes = y_true.shape[1]

    # First compute each class ROC curve and AUROC
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def set_one_ax_auroc(
    ax: Axes,
    fpr: dict,
    tpr: dict,
    roc_auc: dict,
    micro_color,
    macro_color,
    class_colors,
    class_lw,  # ROC 的線條寬度
):
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro (auc = {roc_auc["micro"]:.2f})',
        color=micro_color, linestyle=':', linewidth=4
    )
    ax.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro (auc = {roc_auc["macro"]:.2f})',
        color=macro_color, linestyle=':', linewidth=4
    )
    for i, color in enumerate(class_colors):
        ax.plot(
            fpr[i], tpr[i],
            label=f'Class {i} (auc = {roc_auc[i]:.2f})',
            color=color, lw=class_lw
        )
    ax.plot([0, 1], [0, 1], 'k--', lw=class_lw)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")


def set_one_ax_confusion_matrix(
    fig: Figure,
    ax: Axes,
    matrix: np.ndarray,
    normalized=True,
    font_size=12
):
    classes = np.arange(len(matrix))
    ax_img = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(ax_img, ax=ax)
    ax.set_xticks(classes, classes, fontsize=font_size)
    ax.set_yticks(classes, classes, fontsize=font_size)
    
    fmt = '.2f' if normalized else 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, format(matrix[i, j], fmt),
                verticalalignment='center',
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=font_size
            )

    ax.set_xlabel('Prediction', fontsize=font_size)
    ax.set_ylabel('Ground Truth', fontsize=font_size)


def plot_test_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_name=None,
    save=True
):
    '''`save_name` is default = `model_name`'''
    fpr = dict()  # False Positive Rate
    tpr = dict()  # True Positive Rate
    roc_auc = dict()  # AUC
    
    micro_fpr, micro_tpr, micro_auroc = compute_micro_auroc(y_true, y_pred)
    fpr.update(micro_fpr)
    tpr.update(micro_tpr)
    roc_auc.update(micro_auroc)

    macro_fpr, macro_tpr, macro_auroc = compute_macro_auroc(y_true, y_pred)
    fpr.update(macro_fpr)
    tpr.update(macro_tpr)
    roc_auc.update(macro_auroc)

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f'{model_name.upper()} Result On Testing Set')
    ax1, ax2 = fig.subplots(1, 2)
    ax1: Axes; ax2: Axes
    
    micro_color = 'deeppink'
    macro_color = 'navy'
    class_colors = ['aqua', 'darkorange', 'cornflowerblue','red']
    class_lw = 2

    ax1.set_title('ROC Curve')
    set_one_ax_auroc(
        ax1, fpr, tpr, roc_auc,
        micro_color, macro_color, class_colors, class_lw
    )
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)

    ax2.set_title('ROC Curve (zoom in)')
    set_one_ax_auroc(
        ax2, fpr, tpr, roc_auc,
        micro_color, macro_color, class_colors, class_lw
    )
    ax2.set_xlim(0, 0.2)
    ax2.set_ylim(0.8, 1)

    if save_name is None:
        save_name = model_name
    if save:
        plt.savefig(f'my_code/results/{save_name}_roc.jpg')
    else:
        plt.show()


def plot_test_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_name=None,
    save=True
):
    '''`save_name` is default = `model_name`'''
    matrix = confusion_matrix(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1)
    )

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f'{model_name.upper()} Result On Testing Set')
    ax1, ax2 = fig.subplots(1, 2)
    ax1: Axes; ax2: Axes

    ax1.set_title('Confusion Matrix')
    set_one_ax_confusion_matrix(fig, ax1, matrix, normalized=False)

    recall_m = matrix.astype(np.float32) / matrix.sum(axis=1, keepdims=True)
    ax2.set_title('Confusion Matrix (Recall)')
    set_one_ax_confusion_matrix(fig, ax2, recall_m, normalized=True)

    if save_name is None:
        save_name = model_name
    if save:
        plt.savefig(f'my_code/results/{save_name}_confusion_matrix.jpg')
    else:
        plt.show()


def get_keyframe_positions(
    y_pred: np.ndarray,
    videos_len: Union[np.ndarray, list],
    no_ordering=False
):
    keyframes_list = []

    for video_pred, original_length in zip(y_pred, videos_len):
        effective_video_pred = video_pred[:original_length]
        
        keyframe1_candidates = np.argsort(effective_video_pred[:, 1])[::-1]
        keyframe2_candidates = np.argsort(effective_video_pred[:, 2])[::-1]
        keyframe3_candidates = np.argsort(effective_video_pred[:, 3])[::-1]
        
        if no_ordering:
            keyframe1_idx = keyframe1_candidates[0]
            keyframe2_idx = keyframe2_candidates[0]
            keyframe3_idx = keyframe3_candidates[0]
        
        else:
            for idx in keyframe1_candidates:
                if idx < keyframe2_candidates[0] and idx < keyframe3_candidates[0]:
                    keyframe1_idx = idx
                    break
            for idx in keyframe2_candidates:
                if idx > keyframe1_idx and idx < keyframe3_candidates[0]:
                    keyframe2_idx = idx
                    break
            for idx in keyframe3_candidates:
                if idx > keyframe1_idx and idx > keyframe2_idx:
                    keyframe3_idx = idx
                    break
        
        keyframes_list.append([keyframe1_idx, keyframe2_idx, keyframe3_idx])

    return np.array(keyframes_list)


def compute_avg_position_error(keyframe_pred: np.ndarray, keyframe_true: np.ndarray, model_name: str):
    '''`keyframe_pred` and `keyframe_true` shapes are (N, 3).'''
    pos_error = np.abs(keyframe_pred - keyframe_true)
    m: np.ndarray = pos_error.mean(axis=0)
    df = pd.DataFrame(
        data=np.concatenate([m, m.mean(keepdims=True)])[None, :],
        columns=['keyframe1', 'keyframe2', 'keyframe3', 'avg']
    )
    df[model_name] = 'APE'
    df.set_index(model_name, drop=True, inplace=True)
    return df

