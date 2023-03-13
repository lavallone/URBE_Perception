from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "dataset/URBE_dataset_10000/images"
    annotations_file_path: str = "dataset/URBE_dataset_10000/labels/COCO/annotations.json"
    max_number_images: int = 2500
    num_classes: int = 3 # number of classes in the dataset
    augmentation: bool = False # apply augmentation strategy to input images and bounding boxes
    # by reducing the image size to a multiple of 32, you can get a higher frame rate. Here comes the trade-off between Speed and Accuracy. You can reduce the image size until you receive satisfactory accuracy for your use-case.
    img_size: int = 640 # suggested size of image for YOLOv5 or 416
    img_channels: int = 3 # RGB channels
    batch_size: int = 10 # size of the batches (16 on my local machine)
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # YOLOv5 params
    head: str = "simple" # simple or decoupled
    first_out: int = 48 # 48 for YOLOv5m or 16 for YOLOv5n
    
    # LOSS params - values taken from the official repository code
    weight_class: float = 0.5
    weight_obj: float = 1 
    weight_box: float = 0.05
    
    # TRAIN params
    resume_from_checkpoint: str = None # checkpoint model path from which we want to RESUME the training
    load_pretrained: bool = True # if we want to load pretrained weights (only for the BACKBONE and the NECK)
    lr: float = 2e-4 # learning rate: 2e-4 or 5e-4
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 5e-4 # weight decay as regulation strategy: 5e-4 or 1e-6
    precision: int = 32 # which floating precision to use during training
    
    # PREDICT params - which objects do we want to detect? (trade-off metrics/speed)
    # values taken from https://github.com/AlessandroMondin/YOLOV5m
    nms_iou_thresh: float = 0.6 # nms iou threshold
    conf_threshold: float = 0.01 # first threshold filtering
    
    # LOGGING params
    log_images: int = 4 # how many images to log each time
    log_image_each_epoch: int = 2 # epochs interval we wait to log images
    
    # INFERENCE params
    reduce_inference: bool = False # if we want to prune/quantize the model during training