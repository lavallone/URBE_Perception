import os
import json
from pathlib import Path
from extract import ExtractionToolkit
from pycocotools.coco import COCO
from tqdm import tqdm

def uniqueid():
    seed = 0 # random.getrandbits(20)
    while True:
       yield seed
       seed += 1

def lookup_tables_create():
  
  print("Starting creating the lookup tables...")
  
  coco_waymo = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json")
  coco_bdd100k = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json")
  coco_argoverse = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json")
  images = coco_waymo.dataset["images"] + coco_bdd100k.dataset["images"] + coco_argoverse.dataset["images"]
  old_image_ids = list(map(lambda x: x["id"], images))
  
  waymo = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo"
  bdd100k = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K"
  argoverse = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse"

  id_generator = uniqueid()
  file_images_lookup_table = {}
  img2oldID = {}
  for im in images:
    file_image = ""
    if im["dataset"] == "waymo":
      file_image = waymo + "/images/videos/" + im["file_name"]
      img2oldID["file_image"] = im["id"]
      id = str(next(id_generator))
      file_images_lookup_table["file_image"] = id
      
  # for dir in l:
  #   for v in os.listdir(dir+"/images/videos"):
  #     video_folder = dir+"/images/videos/"+v
  #     for im in os.listdir(video_folder):
  #       file_image = video_folder + "/" + im
  #       id = str(next(id_generator))
  #       file_images_lookup_table[file_image] = id
  
        
  id_generator = uniqueid()
  image_ids_lookup_table = {}
  for old_id in tqdm(old_image_ids):
    id = str(next(id_generator))
    image_ids_lookup_table[old_id] = id

  print(len(file_images_lookup_table.keys()))
  print(len(image_ids_lookup_table.keys()))

  # We also write it in a json file for future uses.
  f = open("/content/drive/MyDrive/VISIOPE/Project/data/img2id.json", "w")
  json.dump(file_images_lookup_table, f)
  f.close()
  f = open("/content/drive/MyDrive/VISIOPE/Project/data/old2newid.json", "w")
  json.dump(image_ids_lookup_table, f)
  f.close()

  return file_images_lookup_table, image_ids_lookup_table

if __name__=="__main__":
    
    img2id = None
    img2oldID = None
    oldID2id = None
    file_path1 = Path("/content/drive/MyDrive/VISIOPE/Project/data/img2id.json")
    file_path2 = Path("/content/drive/MyDrive/VISIOPE/Project/data/img2oldID.json")
    file_path3 = Path("/content/drive/MyDrive/VISIOPE/Project/data/oldID2id.json")
    
    if file_path1.is_file(): # all the files exist
        print("loading the lookup tables...")
        img2id = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/img2id.json"))
        img2oldID = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/img2oldID.json"))
        oldID2id = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/oldID2id.json"))
        print("Done!")
    else: # ne basta uno non caricato
        print("creating the lookup tables...")
        img2id, img2oldID, oldID2id = lookup_tables_create()
        print("Done!")
    
    toolkit = ExtractionToolkit(img2id=img2id, img2oldID=img2oldID, oldID2id=oldID2id)
    #toolkit.extract_images()
    #toolkit.extract_labels()