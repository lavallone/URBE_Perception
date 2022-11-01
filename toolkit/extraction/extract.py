import os
import shutil
import glob
import random
from PIL import Image
from pycocotools.coco import COCO


class ExtractionToolkit:
    def __init__(self, images_lookup_table=None, images_list=None):

        self.images_lookup_table = images_lookup_table
        self.images_list = images_list # all'inizio è 'None'
        
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
        for file_name in self.images_list[:10]:
            id = self.images_lookup_table[file_name]
            n = len(id)
            name = id
            for _ in range(6-n):
                name = '0' + name
            name += '.jpg'
            
            im = Image.open(file_name)
            resized_im = im.resize((1280, 720))
            final_im = resized_im.convert("RGB")
            final_im.save('/content/drive/MyDrive/VISIOPE/Project/data/images/'+ name)

            #shutil.copy(file_name, "/content/drive/MyDrive/VISIOPE/Project/data/images")
            #i = [i for i,c in enumerate(file_name[::-1]) if c=="/"][0]
            #im = file_name[len(file_name)-i:]
            #os.rename("/content/drive/MyDrive/VISIOPE/Project/data/images/"+im, "/content/drive/MyDrive/VISIOPE/Project/data/images/"+id+".jpg")
        print("Done!")
        
    def extract_labels(self):
        print("Starting extracting images...")
        coco_waymo = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json")
        coco_bdd100k = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json")
        coco_argoverse = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json")
        images = coco_waymo.dataset["images"] + coco_bdd100k.dataset["images"] + coco_argoverse.dataset["images"]
        annotations = coco_waymo.dataset["annotations"] + coco_bdd100k.dataset["annotations"] + coco_argoverse.dataset["annotations"] 
        
        print("Retrieving the 'image_ids'...")
        image_ids_list = []
        for file_name in self.images_list:
            for im in images:
                if im["file_name"] == file_name:
                    image_ids_list.append(im["id"])
                    break
        print("Done!")
        
        print("Saving the new annotations files to 'data/labels'...")
        #d = {}
        for file_name, image_id in zip(self.images_list[:10], image_ids_list[:10]):
            value = list( map(lambda y: str(y["category_id"])+" "+str(y["bbox"][0]+" "+str(y["bbox"][1])+" "+str(y["bbox"][2]))+" "+str(y["bbox"][3]), list(filter(lambda x: x["image_id"]==image_id, annotations))) )
            #d[file_name] = value
            id = self.images_lookup_table[file_name]
            f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/"+id+".txt", "w")
            f.write('\n'.join(value))
            f.close
        print("Done!")

            
        #for file_name in self.images_list[:10]:
        #    id = self.images_lookup_table[file_name]
        #    f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/"+id+".txt", "w")
        #    f.write('\n'.join(d[file_name]))
        #    f.close