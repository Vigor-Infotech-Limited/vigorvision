# vigorvision/engine/inference_engine.py
import os
import torch
from tqdm import tqdm
import numpy as np
import cv2

from vigorvision.utils.iou import nms as non_max_suppression
from vigorvision.utils.transforms import preprocess_image
from vigorvision.utils.general import ToTensor
from vigorvision.models.vision.visionmodel import VisionModel
from vigorvision.utils.decoder import PredictionDecoder


class InferenceEngine:
    """
    High-performance inference engine wired directly to the beast PredictionDecoder.
    Keeps all existing behavior (NMS, per-image handling, mixed precision, etc.)
    while using the new decoder for robust multi-scale anchor-aware decoding.
    """

    def __init__(
        self,
        model,
        device=None,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=100,
        use_amp=True,
        return_raw=False,
        decoder_kwargs=None,
    ):
        """
        Args:
            model: Trained VigorVision model (VisionModel).
                   Model must expose anchors/strides/num_classes either via outputs
                   or as attributes (see _resolve_model_detection_params).
            device: torch.device or None.
            conf_thres, iou_thres, max_det: postprocessing thresholds.
            use_amp: use torch.autocast for faster fp16 inference on CUDA.
            return_raw: if True, return raw model outputs (no decode/NMS).
            decoder_kwargs: optional dict passed to PredictionDecoder constructor.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.use_amp = use_amp
        self.return_raw = return_raw

        decoder_kwargs = decoder_kwargs or {}
        # Ensure decoder uses the same conf/iou defaults unless explicitly set
        decoder_kwargs.setdefault("conf_threshold", conf_thres)
        decoder_kwargs.setdefault("iou_threshold", iou_thres)

        self.decoder = PredictionDecoder(**decoder_kwargs)

        # Try to probe model to get anchors, strides, num_classes for faster runtime.
        self._anchors, self._strides, self._num_classes = self._resolve_model_detection_params(self.model)

    def _resolve_model_detection_params(self, model):
        """
        Try multiple common attribute names/placements to find anchors, strides, num_classes.
        Returns (anchors_list, strides_list, num_classes) where anchors_list are tensors on device.
        If anchors/strides cannot be resolved, returns (None, None, num_classes_or_None).
        """
        # 1) If model.forward returns tuple with preds, anchors, strides, we will detect at runtime.
        # 2) Otherwise try common attributes.
        num_classes = getattr(model, "num_classes", None)
        # common names for anchors/strides in detection modules
        possible_anchor_attrs = ["anchors", "anchor_grid", "anchors_grid", "model_anchors"]
        possible_stride_attrs = ["strides", "stride", "strides_list", "model_stride"]

        anchors = None
        strides = None

        # search attributes on model and model.module (if wrapped)
        search_spaces = [model]
        if hasattr(model, "module"):
            search_spaces.append(model.module)

        for space in search_spaces:
            for a in possible_anchor_attrs:
                if hasattr(space, a):
                    anchors = getattr(space, a)
                    break
            for s in possible_stride_attrs:
                if hasattr(space, s):
                    strides = getattr(space, s)
                    break
            if anchors is not None or strides is not None:
                break

        # normalize anchors: convert numpy -> tensor, ensure list of [na,2] tensors on device (float)
        def normalize_anchors(a):
            if a is None:
                return None
            # If provided as one tensor [S, na, 2] or list of arrays/tensors
            if isinstance(a, torch.Tensor):
                # If shape is (S, na, 2)
                if a.ndim == 3:
                    return [x.to(self.device).float() for x in a]
                # else maybe (na,2)
                if a.ndim == 2:
                    return [a.to(self.device).float()]
            if isinstance(a, (list, tuple)):
                out = []
                for x in a:
                    if isinstance(x, np.ndarray):
                        out.append(torch.tensor(x, device=self.device).float())
                    elif isinstance(x, torch.Tensor):
                        out.append(x.to(self.device).float())
                    else:
                        out.append(torch.tensor(np.asarray(x), device=self.device).float())
                return out
            # fallback: try numpy
            try:
                arr = np.asarray(a)
                if arr.ndim == 3:
                    return [torch.tensor(arr[i], device=self.device).float() for i in range(arr.shape[0])]
                if arr.ndim == 2:
                    return [torch.tensor(arr, device=self.device).float()]
            except Exception:
                return None

        anchors_norm = normalize_anchors(anchors)

        # normalize strides
        if strides is not None:
            if isinstance(strides, torch.Tensor):
                strides_norm = strides.tolist() if strides.ndim == 1 else [int(s.item()) for s in strides.flatten()]
            elif isinstance(strides, (list, tuple, np.ndarray)):
                strides_norm = [int(s) for s in strides]
            else:
                strides_norm = None
        else:
            strides_norm = None

        # If we found anchors but not strides, try to infer strides from anchors if possible (rare)
        return anchors_norm, strides_norm, num_classes

    def _attempt_runtime_unwrap(self, preds_output):
        """
        If model returns (preds, anchors, strides, ...) at forward time, extract them.
        This handles models that dynamically produce anchor/stride info with predictions.
        """
        if isinstance(preds_output, (list, tuple)) and len(preds_output) >= 1:
            # try to find a predictions tensor/list first
            preds = preds_output[0]
            anchors = None
            strides = None
            # heuristics: second item could be anchors, third strides
            if len(preds_output) >= 3:
                anchors = preds_output[1]
                strides = preds_output[2]
            else:
                # search objects for anchors/strides
                for item in preds_output[1:]:
                    if isinstance(item, (list, tuple)):
                        # maybe anchors list
                        anchors = item
                    elif isinstance(item, torch.Tensor) and item.ndim in (1, 2, 3):
                        # could be stride vector or anchor tensor
                        if item.ndim == 1:
                            strides = item
                        elif item.ndim == 3:
                            anchors = item
            return preds, anchors, strides
        return preds_output, None, None

    @torch.no_grad()
    def predict(self, images):
        """
        Run inference on a batch (list) of images.

        Args:
            images: list of numpy arrays / filepaths / torch tensors.

        Returns:
            List of NMS-filtered detections per image (same format as before).
        """
        processed_imgs = []
        orig_shapes = []

        for img in images:
            img_tensor, orig_shape = preprocess_image(img)  # must return tensor normalized to model input and orig_shape (h,w)
            processed_imgs.append(img_tensor)
            orig_shapes.append(orig_shape)

        batch = torch.stack(processed_imgs).to(self.device)
        input_h, input_w = batch.shape[2], batch.shape[3]

        # Forward
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            model_output = self.model(batch)

        # If user requested raw outputs, return them immediately
        if self.return_raw:
            return model_output

        # If model returned preds + anchors/strides in forward, capture them
        preds, anchors_from_forward, strides_from_forward = self._attempt_runtime_unwrap(model_output)

        # Prefer runtime-forward anchors/strides, else fall back to pre-probed ones
        anchors = None
        strides = None

        if anchors_from_forward is not None:
            # normalize anchors_from_forward to list of tensors on device
            anchors = anchors_from_forward
            if not isinstance(anchors, list):
                # try to convert tensor->list
                if isinstance(anchors, torch.Tensor) and anchors.ndim == 3:
                    anchors = [anchors[i].to(self.device).float() for i in range(anchors.shape[0])]
        else:
            anchors = self._anchors

        if strides_from_forward is not None:
            if isinstance(strides_from_forward, torch.Tensor):
                if strides_from_forward.ndim == 1:
                    strides = [int(x.item()) for x in strides_from_forward]
                else:
                    strides = [int(x) for x in strides_from_forward.flatten()]
            elif isinstance(strides_from_forward, (list, tuple, np.ndarray)):
                strides = [int(x) for x in strides_from_forward]
            else:
                strides = None
        else:
            strides = self._strides

        num_classes = getattr(self.model, "num_classes", None) or self._num_classes
        if num_classes is None:
            raise AttributeError(
                "Unable to determine num_classes. Ensure model has `num_classes` attribute or "
                "that your model.forward() returns (preds, anchors, strides, ...)."
            )

        # Final safety: anchors and strides are required by the PredictionDecoder
        if anchors is None or strides is None:
            raise AttributeError(
                "Anchors and strides not found. Model must expose anchors/strides either as forward outputs "
                "or as attributes (e.g., model.anchors, model.strides)."
            )

        # Ensure anchors are tensors on the correct device
        normed_anchors = []
        if isinstance(anchors, (list, tuple)):
            for a in anchors:
                if isinstance(a, np.ndarray):
                    normed_anchors.append(torch.tensor(a, device=self.device).float())
                elif isinstance(a, torch.Tensor):
                    normed_anchors.append(a.to(self.device).float())
                else:
                    normed_anchors.append(torch.tensor(np.asarray(a), device=self.device).float())
        else:
            # single tensor -> try to expand to list by first dim
            if isinstance(anchors, torch.Tensor) and anchors.ndim == 3:
                normed_anchors = [anchors[i].to(self.device).float() for i in range(anchors.shape[0])]
            else:
                raise TypeError("Unexpected anchors format. Expect list-of-(na,2) tensors or a (S,na,2) tensor.")

        # Run decoder:
        # PredictionDecoder expects: (predictions, anchors, strides, num_classes)
        decoded_batch = self.decoder(preds, normed_anchors, strides, num_classes)

        results = []
        for i_img, pred in enumerate(decoded_batch):
            # pred is [N, 6] tensor: x1,y1,x2,y2,score,cls (in model input pixel coords)
            if pred.shape[0] == 0:
                results.append([])
                continue

            # Rescale boxes from model input size -> original image size (best-effort)
            orig_h, orig_w = orig_shapes[i_img]
            # compute simple scale factors
            scale_x = orig_w / float(input_w)
            scale_y = orig_h / float(input_h)

            boxes = pred[:, :4].clone()
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

            pred_rescaled = torch.cat([boxes, pred[:, 4:]], dim=1)

            # Apply your existing NMS util (expects prediction in same format)
            nms_output = non_max_suppression(
                prediction=pred_rescaled,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det,
            )

            results.append(nms_output)

        return results


# lightweight test harness (keeps your original interface)
def test_inference_engine():
    import numpy as np

    # Dummy input: 2 blank images
    dummy_imgs = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(2)]

    # Load model (random weights for test)
    model = VisionModel(num_classes=11)
    model.eval()

    engine = InferenceEngine(model=model, conf_thres=0.25, iou_thres=0.45, return_raw=False)
    preds = engine.predict(dummy_imgs)

    for i, p in enumerate(preds):
        print(f"Image {i+1}: {p}")


if __name__ == "__main__":
    test_inference_engine()
