import argparse
import os
import json
import waymo
import bdd100k
from pycocotools.coco import COCO

def add_timeofday():

  print("Building 'timeofday_list'...")
  d1 = json.load(open("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/det_train.json"))
  d2 = json.load(open("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/det_val.json"))
  
  video_list = os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos")
  timeofday_list = []
  for e in d1:
    if e["name"][:-4] in video_list:
      timeofday_list.append({"video_name" : e["name"][:-4], "timeofday" : e["attributes"]["timeofday"]})
  for e in d2:
    if e["name"][:-4] in video_list:
      timeofday_list.append({"video_name" : e["name"][:-4], "timeofday" : e["attributes"]["timeofday"]})

  print("Done!")
  return timeofday_list

def clean_json(coco_train, coco_val, d, lookup_video):
    d["categories"] = [{"name" : "vehicle", "id" : 0}, {"name" : "person", "id" : 1}, {"name" : "motorbike", "id" : 2}]
    
    for img in coco_train.dataset["images"]:
        img["video_id"] = lookup_video[img["sid"]]
        img["file_name"] = img["video_id"] + "/" + img["name"]
        img["dataset"] = "argoverse"
        img["timeofday"] = None
        img.pop("sid",None)
        img.pop("fid",None)
        img.pop("name",None)
    image_list = coco_train.dataset["images"]
    start_id = len(image_list)
    for img in coco_val.dataset["images"]:
        img["id"] = img["id"] + start_id
        start_id = start_id + 1
        img["video_id"] = lookup_video[img["sid"]]
        img["file_name"] = img["video_id"] + "/" + img["name"]
        img["dataset"] = "argoverse"
        img["timeofday"] = None
        img.pop("sid",None)
        img.pop("fid",None)
        img.pop("name",None)
    image_list = image_list + coco_val.dataset["images"]
    d["images"] = image_list
    
    ann_ids=[]
    for ann in coco_train.dataset["annotations"]:
        if ann["iscrowd"]==False:
            if ann["category_id"]==0 or ann["category_id"]==2 or ann["category_id"]==3 or ann["category_id"]==4 or ann["category_id"]==5:
                ann.pop("area",None)
                ann.pop("ignore",None)
                ann.pop("track",None)
                ann.pop("iscrowd",None)
                if ann["category_id"]==2 or ann["category_id"]==4 or ann["category_id"]==5:
                    ann["category_id"] = 0
                elif ann["category_id"]==0:
                    ann["category_id"] = 1
                elif ann["category_id"]==3:
                    ann["category_id"] = 2
                ann_ids.append(ann["id"])
            else:
                continue
    ann_list = coco_train.loadAnns(ann_ids)
    ann_ids=[]
    for ann in coco_val.dataset["annotations"]:
        if ann["iscrowd"]==False:
            if ann["category_id"]==0 or ann["category_id"]==2 or ann["category_id"]==3 or ann["category_id"]==4 or ann["category_id"]==5:
                ann.pop("area",None)
                ann.pop("ignore",None)
                ann.pop("track",None)
                ann.pop("iscrowd",None)
                if ann["category_id"]==2 or ann["category_id"]==4 or ann["category_id"]==5:
                    ann["category_id"] = 0
                elif ann["category_id"]==0:
                    ann["category_id"] = 1
                elif ann["category_id"]==3:
                    ann["category_id"] = 2
                ann_ids.append(ann["id"])
            else:
                continue
    ann_list = ann_list + coco_val.loadAnns(ann_ids)
    d["annotations"] = ann_list

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        tfrecord_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/tfrecord"
        images_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json"
        
        toolkit = waymo.WaymoToolKit(tfrecord_dir=tfrecord_dir, images_dir=images_dir, labels_json=labels_json, image_or_label="image")
        toolkit.waymo_building()
        
    elif args.dataset == "bdd100k":
        labels_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/old_json"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json"
        timeofday_list = add_timeofday()
        
        toolkit = bdd100k.BDD100KToolKit(labels_dir=labels_dir, labels_json=labels_json, timeofday_list = timeofday_list)
        toolkit.bdd100k_building()
        
    elif args.dataset == "argoverse": # since the labels are COCO-like, we just need to clean the already existed json file!
        images_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos"
        old_train_labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/old_train.json"
        old_val_labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/old_val.json"
        labels_json = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json"
        
        d = {}
        coco_train = COCO(old_train_labels_json)
        coco_val = COCO(old_val_labels_json)
        list_videos = coco_train.dataset["sequences"] 
        list_videos = list_videos + coco_val.dataset["sequences"]
        d["info"] = coco_train.dataset["info"]
        #d["videos"] = []
        lookup_video = {}
        for i, name_video in enumerate(list_videos):
            lookup_video[i] = name_video
            totalFrames = 0
            for f in os.listdir(images_dir+"/"+name_video):
                totalFrames = totalFrames + 1
            #d["videos"].append({"id" : name_video, "num_frames" : totalFrames, "time" : None})
        
        # Ora che abbiamo agggiunto la sezione dei video al file 'json', possiamo iniziare a pulirlo un po'...
        print("cleaning 'old_train.json' and 'old_val.json'...") 
        clean_json(coco_train, coco_val, d, lookup_video)
        print("Done!")
        print("copying to 'annotations.json'...")
        json.dump(d, open(labels_json, "w"))
        print("Done!")
    else:
        pass