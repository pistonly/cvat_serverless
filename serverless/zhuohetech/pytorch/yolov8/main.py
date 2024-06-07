import json
import base64
from PIL import Image
import io
import torch
import sys
print(sys.path)
from ultralytics import YOLO
import yaml


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # Read the DL model
    model = YOLO("/opt/nuclio/common/resource/yolov8/yolov8l.pt")
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
    result = context.user_data.model.predict(image)[0].numpy()
    boxes = result.boxes
    encoded_results = []

    for box, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        encoded_results.append({
            'confidence':
            conf,
            'label':
            context.user_data.labels[int(cls)],
            'points':
            box,
            'type':
            'rectangle'
        })

    print(encoded_results)
    return context.Response(body=json.dumps(encoded_results),
                            headers={},
                            content_type='application/json',
                            status_code=200)
