import os
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from .box_ops import xywh_to_xyxy
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Optional

# Ensure consistent plots
sns.set(style="whitegrid", palette="muted", font_scale=1.2)


def plot_loss_curve(train_loss, val_loss, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_loss, label='Val Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_metrics(history, save_dir):
    """Plot training metrics: loss, precision, recall, mAP."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key in ['loss', 'precision', 'recall', 'mAP']:
        plt.figure(figsize=(8, 5))
        plt.plot(history[key], label=f'{key.capitalize()}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(key.capitalize())
        plt.title(f'{key.upper()} over Epochs')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{key}.png"), bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_scores, class_names, save_path=None):
    """Plot Precision-Recall curve for each class."""
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP={ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_predictions(image, boxes, scores, labels, class_names, save_path=None, score_thresh=0.3):
    """
    Visualize predictions on an image.
    Args:
        image: np.array (H, W, 3) in BGR format.
        boxes: Tensor[N, 4] in xywh or xyxy format
        scores: Tensor[N]
        labels: Tensor[N]
        class_names: list of class strings
        save_path: path to save image
        score_thresh: threshold for confidence
    """
    image = image.copy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    if boxes.shape[1] == 4 and not np.all(boxes[:, 2] > boxes[:, 0]):
        boxes = xywh_to_xyxy(boxes)

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(image, label_text, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if save_path:
        cv2.imwrite(save_path, image)
    return image


# Default colors (up to 80 classes) — cycling
COLORS = np.random.default_rng(42).integers(0, 255, size=(80, 3), dtype=np.uint8)

def plot_boxes(
    img: Union[np.ndarray, torch.Tensor],
    boxes: Union[np.ndarray, torch.Tensor],
    scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    color_map: Optional[np.ndarray] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    alpha: float = 0.4,
    draw_labels: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    resize_info: Optional[Tuple[Tuple[int, int], float, Tuple[int, int]]] = None
):
    """
    Draws bounding boxes with optional class labels and confidences on an image.

    Args:
        img: Image array (HWC, RGB) or torch.Tensor (CHW) in uint8 or float (0-1)
        boxes: [N, 4] in xyxy format
        scores: [N] confidence scores (optional)
        labels: [N] class indices (optional)
        class_names: list of class names (optional)
        color_map: custom color map array [num_classes, 3]
        line_thickness: thickness of bounding box lines
        font_scale: font size for labels
        alpha: transparency of filled rectangle behind text
        draw_labels: whether to draw labels and scores
        save_path: file path to save the image
        show: whether to display image in a window
        resize_info: tuple ((orig_h, orig_w), ratio, (pad_x, pad_y)) for reversing letterbox
    Returns:
        img_out: Image with boxes as np.ndarray (HWC, RGB)
    """
    # Convert torch → numpy
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # Copy image to draw on
    img_out = img.copy()

    # Reverse letterbox resize if info provided
    if resize_info:
        (orig_h, orig_w), ratio, (pad_x, pad_y) = resize_info
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= ratio
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_h)

    # Convert torch → numpy for boxes/scores/labels
    for arr_name in ["boxes", "scores", "labels"]:
        arr = locals()[arr_name]
        if isinstance(arr, torch.Tensor):
            locals()[arr_name] = arr.cpu().numpy()

    boxes = boxes.astype(np.float32)
    num_boxes = len(boxes)
    if num_boxes == 0:
        return img_out

    # Color palette
    colors = color_map if color_map is not None else COLORS

    for i in range(num_boxes):
        x1, y1, x2, y2 = boxes[i]
        cls_id = int(labels[i]) if labels is not None else 0
        color = tuple(int(c) for c in colors[cls_id % len(colors)])

        # Draw box
        cv2.rectangle(img_out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=line_thickness)

        # Draw label
        if draw_labels:
            label_text = ""
            if labels is not None and class_names and 0 <= cls_id < len(class_names):
                label_text = class_names[cls_id]
            elif labels is not None:
                label_text = f"{cls_id}"
            if scores is not None:
                label_text += f" {scores[i]:.2f}"

            if label_text:
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                th = int(th * 1.2)
                cv2.rectangle(img_out, (int(x1), int(y1) - th - baseline), (int(x1) + tw, int(y1)), color, -1)
                overlay = img_out.copy()
                cv2.rectangle(overlay, (int(x1), int(y1) - th - baseline), (int(x1) + tw, int(y1)), color, -1)
                img_out = cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0)
                cv2.putText(
                    img_out, label_text, (int(x1), int(y1) - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
                )

    # Save image
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))

    # Show image
    if show:
        cv2.imshow("Detections", cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_out


def test_plot_boxes():
    # Dummy test
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    boxes = np.array([[50, 50, 200, 200], [300, 100, 450, 400]])
    scores = np.array([0.88, 0.76])
    labels = np.array([0, 1])
    class_names = ["cat", "dog"]

    out = plot_boxes(img, boxes, scores, labels, class_names, show=True)
    print("Plot test completed.")


if __name__ == "__main__":
    test_plot_boxes()
