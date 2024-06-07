import json
import base64
from PIL import Image
import io
import torch
import yaml


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custo
    model = torch.hub.load(
        "/opt/nuclio/common/resource/yolov5",
        "custom",
        path="/opt/nuclio/common/resource/yolov5/yolov5x.pt",
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
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    yolo_results_json = context.user_data.model(
        image).pandas().xyxy[0].to_dict(orient='records')

    encoded_results = []
    for result in yolo_results_json:
        encoded_results.append({
            'confidence':
            result['confidence'],
            'label':
            context.user_data.labels[int(result['class'])],
            'points':
            [result['xmin'], result['ymin'], result['xmax'], result['ymax']],
            'type':
            'rectangle'
        })

    return context.Response(body=json.dumps(encoded_results),
                            headers={},
                            content_type='application/json',
                            status_code=200)
