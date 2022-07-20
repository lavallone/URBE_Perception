import os
import json
from pathlib import Path
from extract import ExtractionToolkit

def uniqueid():
    seed = 0 #random.getrandbits(20)
    while True:
       yield seed
       seed += 1

def images_lookup_table_create():
    
  waymo = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo"
  bdd100k = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K"
  argoverse = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse"

  id_generator = uniqueid()
  l = [waymo, bdd100k, argoverse]
  images_lookup_table = {}
  file_name_list = []
  for dir in l:
    for v in os.listdir(dir+"/images/videos"):
      video_folder = dir+"/images/videos/"+v
      for im in os.listdir(video_folder):
        file_image = video_folder + "/" + im
        id = str(next(id_generator))
        file_name_list.append(file_image)
        images_lookup_table[file_image] = id

  print(len(images_lookup_table.keys()))

  # We also write it in a json file for future uses.
  f = open("/content/drive/MyDrive/VISIOPE/Project/data/images_lookup_table.json", "w")
  json.dump(images_lookup_table, f)
  f.close()

  return images_lookup_table

if __name__=="__main__":
    
    images_lookup_table = None
    file_path = Path("/content/drive/MyDrive/VISIOPE/Project/data/images_lookup_table.json")
    if False:#file_path.is_file(): # if the file exists
        print("loading 'images_lookup_table'...")
        images_lookup_table = json.load(open("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/old_train.json"))
    else:
        print("creating 'images_lookup_table'...")
        images_lookup_table = images_lookup_table_create()
        print("Done!")
    toolkit = ExtractionToolkit(images_lookup_table=images_lookup_table)
    toolkit.extract_images()
    toolkit.extract_labels()