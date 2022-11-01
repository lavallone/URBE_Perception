import os
import glob
import random
from PIL import Image
import json
from pycocotools.coco import COCO
from tqdm import tqdm

def uniqueid():
    seed = 0 #random.getrandbits(20)
    while True:
       yield seed
       seed += 1

def name_id(id, x):
    n = len(id)
    name = id
    for _ in range(x-n):
        name = '0' + name
    return name

class ExtractionToolkit:
    def __init__(self, file_images_lookup_table=None, image_ids_lookup_table=None, images_list=None, old_ids_list=None):

        self.file_images_lookup_table = file_images_lookup_table
        self.image_ids_lookup_table = image_ids_lookup_table
        self.images_list = images_list # all'inizio è 'None'
        self.old_ids_list = old_ids_list # all'inizio è 'None'
        
    def extract_images(self):
        print("Starting extracting images...")
        
        # first of all, we delete the previous images inside the folder
        print("Deleting the previous images...")
        for f in glob.glob('{}/*.jpg'.format("/content/drive/MyDrive/VISIOPE/Project/data/images"), recursive=True):
            os.remove(f)
        print("Done!")
        
        # we select images because many of them are similar (subsequent frame images)
        waymo_list = []
        bdd100k_list = []
        argoverse_list = []
        
        print("Processing Waymo images...")
        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos/"+v
            images_list = sorted(os.listdir(video_folder))
            for i in range(0,len(images_list), 6):
                waymo_list = waymo_list + [video_folder+"/"+images_list[i], video_folder+"/"+images_list[i+1], video_folder+"/"+images_list[i+2]]
        
        print("Processing BDD100K images...")
        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"+v+"/"
            images_list = sorted(os.listdir(video_folder))
            for i in range(0, len(images_list), 2):
                bdd100k_list.append(video_folder+images_list[i])
                
        print("Processing Argoverse images...")
        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"+v+"/"
            images_list = sorted(os.listdir(video_folder))
            for i in range(0, len(images_list), 3):
                argoverse_list.append(video_folder+images_list[i])
                
        self.images_list = waymo_list + bdd100k_list + argoverse_list
        random.shuffle(self.images_list) # for shuffling the order of the images
        print("Now the images are: {}".format(len(self.images_list)))
        
        print("Saving the new images to 'data/images'...")
        for file_name in tqdm(self.images_list[:10]):
            id = self.file_images_lookup_table[file_name]
            name = name_id(id, 6)
            name += '.jpg'
            
            im = Image.open(file_name)
            resized_im = im.resize((1280, 720))
            final_im = resized_im.convert("RGB")
            final_im.save('/content/drive/MyDrive/VISIOPE/Project/data/images/'+ name)
        print("Done!")
        
    def extract_labels(self):
        print("Starting extracting labels...")
        coco_waymo = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json")
        coco_bdd100k = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json")
        coco_argoverse = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json")
        images = coco_waymo.dataset["images"] + coco_bdd100k.dataset["images"] + coco_argoverse.dataset["images"]
        annotations = coco_waymo.dataset["annotations"] + coco_bdd100k.dataset["annotations"] + coco_argoverse.dataset["annotations"] 
        
        new_annotations = {"info" : {"num_images" : 191723}, "images" : [], "annotations" : []}
        
        print("Retrieving the old 'image_ids'...")
        image_ids_list = []
        for file_name in tqdm(self.images_list[:10]):
            d = {}
            im = list(filter(lambda x: x["file_name"]==file_name, images))[0]
            print(im)
            #for im in images:
                #if im["file_name"] == file_name:
                    
            id = self.image_ids_lookup_table[file_name]
            id = name_id(id, 6)
            image_ids_list.append(im["id"])
            d["id"] = id
            d["file_name"] = file_name
            d["width"] = 1280
            d["height"] = 720
            d["timeofday"] = im["timeofday"]
            new_annotations["images"].append(d)
            #break
        print("Done!")
        
        print("Saving the new annotations files to 'data/labels'...")
        #d = {}
        for file_name,image_id in tqdm(zip(self.images_list[:10], image_ids_list[:10])):
            annot = list(filter(lambda x: x["image_id"]==image_id, annotations))
            for ann in annot:
                new_id = self.image_ids_lookup_table[file_name]
                new_id = name_id(new_id, 6)
                ann["image_id"] = new_id
                new_annotations["annotations"].append(ann)
        
        # number of annotations
        print("Total number of annotations: "+ len(new_annotations["annotations"]))
        
        # standardizziamo e unifichiamo gli ID delle annotazioni
        print("Setting annotations IDs to be unique")
        id_generator = uniqueid()
        for ann in new_annotations["annotations"]:
            id = str(next(id_generator))
            id = name_id(id, 6) # dobbiamo capire quante annotations per capire quanti zeri aggiungere!
            ann["id"] = id
        
        print("Writing the 'annotations.json' file...")
        f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/annotations.json", "w")
        json.dump(new_annotations, f)
        f.close()
        print("Done!")