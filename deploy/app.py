from yolo_onnx.yolov8_onnx import YOLOv8
import json
import base64
from io import BytesIO
from PIL import Image

# Definec a global parameter for the Initialized YOLOv8 object detector
# global parameter: remains in memory during the lifetime of the function; not be re-initialized for each function all
yolo_detector = YOLOv8('model.onnx')


def main(event, context):

    # Get request body
    body = json.loads(event['body'])

    # get arguments
    image = body['image']               # encoded image
    size = body.get('size', 640)        # size
    conf_thres = body.get('conf_thres', 0.3)        # confidence threshold
    iou_thres = body.get('iou_thres', 0.5)          # IOU threshold

    # decoded
    img = Image.open(
        BytesIO(
            base64.b64decode(image.encode('ascii'))
        ))

    # predict
    pred_yolo = yolo_detector(
        img, size=size, conf_thres=conf_thres, iou_thres=iou_thres)

    # return prediction
    return {
        "statusCode": 200,
        "body": json.dumps({
            "detections": pred_yolo
        }),
    }

# return sample:
# [
#   {"bbox": [0, 264, 473, 1351], "score": 0.932, "class_id": 0},
#   {"bbox": [1286, 199, 1918, 1353], "score": 0.93, "class_id": 0},
#   {"bbox": [1047, 288, 1534, 1348], "score": 0.895, "class_id": 0},
#   {"bbox": [494, 433, 943, 1357], "score": 0.895, "class_id": 0},
#   {"bbox": [1021, 748, 1111, 901], "score": 0.679, "class_id": 41}
# ]
