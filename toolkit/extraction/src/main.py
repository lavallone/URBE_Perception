import argparse
import waymo
import bdd100k

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        tfrecord_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/tfrecord"
        images_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/videos"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/train/train.json"
        
        toolkit = waymo.WaymoToolKit(tfrecord_dir=tfrecord_dir, images_dir=images_dir, labels_json=labels_json, image_or_label="image")
        toolkit.waymo_extraction()
    elif args.dataset == "bdd100k":
        labels_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/train"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/train/train.json"
        
        toolkit = bdd100k.BDD100KToolKit(labels_dir=labels_dir, labels_json=labels_json)
        toolkit.bdd100k_extraction()
    else:
        pass