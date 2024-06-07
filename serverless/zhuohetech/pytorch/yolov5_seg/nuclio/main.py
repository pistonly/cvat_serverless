import json
import base64
from PIL import Image
import io
import torch
import numpy as np
from main_utils import (letterbox, non_max_suppression, process_mask,
                        mask2segment, to_cvat_mask, process_mask_native,
                        scale_boxes)
import yaml
import cv2


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custo
    model = torch.hub.load(
        "/opt/nuclio/common/resource/yolov5",
        "custom",
        path="/opt/nuclio/common/resource/yolov5/yolov5x-seg.pt",
        source="local",
        trust_repo=True,
        skip_validation=True)

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    context.user_data.model = model
    context.user_data.labels = labels

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("run yolo-v5-seg model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = np.array(Image.open(buf))

    h0, w0 = image.shape[:2]
    image, r, (dw, dh) = letterbox(image)
    h, w = image.shape[:2]
    image = image.transpose((2, 0, 1))  # CHW
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float() / 255
    image = image[None]

    pred, proto = context.user_data.model(image)[:2]
    pred = non_max_suppression(pred, 0.25, 0.65, nm=32)

    # retina
    det = pred[0]
    det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], (h0, w0)).round()
    mask = process_mask_native(proto[0], det[:, 6:], det[:, :4], (h0, w0))
    det = det.cpu().numpy()

    results = []
    for i, mask_i in enumerate(mask):
        mask_i = mask_i.cpu().numpy().astype(np.uint8)
        segment_i = mask2segment(mask_i)
        Xmin = int(np.min(segment_i[:, 0]))
        Xmax = int(np.max(segment_i[:, 0]))
        Ymin = int(np.min(segment_i[:, 1]))
        Ymax = int(np.max(segment_i[:, 1]))
        cvat_mask = to_cvat_mask((Xmin, Ymin, Xmax, Ymax), mask_i)
        results.append({
            "confidence": str(det[i][4]),
            "label": context.user_data.labels[int(det[i][5])],
            "points": segment_i.ravel().tolist(),
            "mask": cvat_mask,
            "type": "mask",
        })

    return context.Response(body=json.dumps(results),
                            headers={},
                            content_type='application/json',
                            status_code=200)
