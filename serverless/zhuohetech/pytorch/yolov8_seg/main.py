import json
import base64
from PIL import Image
import io
import torch
import sys
print(sys.path)
from ultralytics import YOLO
import yaml
import numpy as np
import cv2


def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def mask2segment(x, strategy='largest'):
    c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if c:
        if strategy == 'concat':  # concatenate all segments
            c = np.concatenate([x.reshape(-1, 2) for x in c])
        elif strategy == 'largest':  # select largest segment
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
    else:
        c = np.zeros((0, 2))  # no segments found
    return c.astype(np.float32)

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # Read the DL model
    model = YOLO("/opt/nuclio/common/resource/yolov8/yolov8x-seg.pt")
    context.user_data.model = model
    context.user_data.labels = labels

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    result = context.user_data.model.predict(image, retina_masks=True)[0].numpy()
    boxes = result.boxes
    masks = result.masks
    encoded_results = []

    for mask_i, conf_i, cls_i in zip(masks.masks, boxes.conf.tolist(), boxes.cls.tolist()):
        mask_i = mask_i.astype(np.uint8)
        segment_i = mask2segment(mask_i)
        Xmin = int(np.min(segment_i[:, 0]))
        Xmax = int(np.max(segment_i[:, 0]))
        Ymin = int(np.min(segment_i[:, 1]))
        Ymax = int(np.max(segment_i[:, 1]))
        cvat_mask = to_cvat_mask((Xmin, Ymin, Xmax, Ymax), mask_i)
        encoded_results.append({
            "confidence": conf_i,
            "label": context.user_data.labels[int(cls_i)],
            "points": segment_i.ravel().tolist(),
            "mask": cvat_mask,
            "type": "mask",
        })
    return context.Response(body=json.dumps(encoded_results),
                            headers={},
                            content_type='application/json',
                            status_code=200)
