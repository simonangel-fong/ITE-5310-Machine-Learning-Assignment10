from ultralytics import YOLO

YOLO_V8N = "./model/base/yolov8n.pt"  # the model to train customized model
YAML_FILE_PATH = "datasets/config.yaml"  # path to data file
NUM_EPOCH = 1  # number of epochs to train for, at leat 300 epochs
NUM_PATIENCE = 1  # patience for early stop


def YOLO_train(train_model, yaml_path, epochs_num, patience_num):
    '''Train Customized Model'''
    model = YOLO(train_model)  # train from the existing model
    model.train(
                data=yaml_path,
                epochs=epochs_num,
                patience=patience_num)


YOLO_train(train_model=YOLO_V8N,
           yaml_path=YAML_FILE_PATH,
           epochs_num=NUM_EPOCH,
           patience_num=NUM_PATIENCE)
