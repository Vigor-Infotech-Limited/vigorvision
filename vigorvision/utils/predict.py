import os
import cv2
import torch
import numpy as np
from vigorvision.utils.decoder import PredictionDecoder
from vigorvision.utils.box_ops import non_max_suppression
from vigorvision.utils.general import set_seed, xywh2xyxy, load_image
from vigorvision.utils.plotting import plot_boxes
from vigorvision.models.build import build_model
from vigorvision.utils.transforms import val_preprocess


@torch.no_grad()
def run_inference(
    model,
    image_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    conf_thresh=0.25,
    iou_thresh=0.45,
    save_result=False,
    save_dir="predictions/",
    class_names=None,
):
    model.eval().to(device)

    img, original, scale = load_image(image_path)
    tensor = val_preprocess(img).unsqueeze(0).to(device)

    preds = model(tensor)
    preds = PredictionDecoder(preds)

    detections = non_max_suppression(preds, conf_thresh, iou_thresh)[0]

    if detections is not None:
        detections[:, :4] = detections[:, :4] / scale  # Rescale boxes
        annotated = plot_boxes(original, detections, class_names)

        if save_result:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, annotated)
        return annotated, detections
    else:
        return original, torch.empty((0, 6))


def predict_folder(
    model,
    folder_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    conf_thresh=0.25,
    iou_thresh=0.45,
    save_result=True,
    save_dir="predictions/",
    class_names=None
):
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    results = []
    for path in image_paths:
        _, dets = run_inference(
            model, path, device, conf_thresh, iou_thresh, save_result, save_dir, class_names
        )
        results.append((path, dets))
    return results


def test_predict_single_image():
    config = {
        "model_type": "visionmodel",
        "weights_path": "weights/best.pt",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load checkpoint and model ===
    checkpoint = torch.load(config["weights_path"], map_location=device)
    model = build_model(config["model_type"], num_classes=len(checkpoint.get("class_names", [])))
    model.load_state_dict(checkpoint["model_state_dict"])

    # === Get class names from checkpoint ===
    class_names = checkpoint.get("class_names", None)
    if class_names is None:
        raise ValueError("No class names found in checkpoint. Please re-train with class_names saved.")

    annotated, detections = run_inference(
        model=model,
        image_path="sample.jpg",
        device=device,
        conf_thresh=0.3,
        iou_thresh=0.5,
        save_result=True,
        save_dir="predictions/",
        class_names=class_names,
    )
    print(f"Detections: {detections}")


if __name__ == "__main__":
    test_predict_single_image()
