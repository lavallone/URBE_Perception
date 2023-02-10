from dataclasses import dataclass


@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "dataset/URBE_dataset/images"
    annotations_file_path: str = "dataset/URBE_dataset/labels/COCO/annotations.json"
    max_number_images: int = 150#3500
    num_classes: int = 3 # number of classes in the dataset
    augmentation: bool = False # apply augmentation strategy to input images and bounding boxes
    img_size: int = 640  # suggested size of image for YOLOv5
    img_channels: int = 3 # RGB channels
    batch_size: int = 1 # size of the batches
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # YOLOv5 params
    head: str = "simple" # or decoupled
    first_out: int = 48 # for YOLOv5m or 16 for YOLOv5n
    
    # LOSS params
    weight_class: float = 0.5
    weight_obj: float = 1 
    weight_box: float = 0.05
    ignore_iou_thresh: float = 0.5
    
    # TRAIN params
    resume_from_checkpoint: str = None # checkpoint model path from which we want to RESUME the training
    load_pretrained: bool = True
    lr: float = 5e-4 # 2e-4 or 5e-4
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 5e-4 #1e-6 # weight decay as regulation strategy
    
    # PREDICT params
    nms_iou_thresh: float = 0.6
    conf_threshold: float = 0.25 #.01 to get all possible bboxes, trade-off metrics/speed --> we choose metrics
    
    # LOGGING params
    log_images: int = 4 # how many images to log each time
    log_image_each_epoch: int = 2 # epochs interval we wait to log images