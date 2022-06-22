import argparse
import waymo

if __name__=="main":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default="waymo")
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        tfrecord_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/prova"
        images_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/videos"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/train/train.json"
        
        toolkit = waymo.WaymoToolKit(tfrecord_dir=tfrecord_dir, images_dir=images_dir, labels_json=labels_json, image_or_label="image")
        toolkit.waymo_extraction()
    else:
        print("porco")
        pass