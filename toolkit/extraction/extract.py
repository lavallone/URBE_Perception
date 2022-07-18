import os
import shutil
from pycocotools.coco import COCO


class ExtractionToolkit:
    def __init__(self, images_lookup_table=None, images_list=None, image_or_label=None, porco=None):

        self.images_lookup_table = images_lookup_table
        #self.ids_list = ids_listtt
        self.porco=porco
        self.images_list = images_list # all'inizio è 'None'
        self.image_or_label = image_or_label
        
    def extract_images(self):
        # we select images because many of them are similar (subsequent frame images)
        waymo_list = []
        bdd100k_list = []
        argoverse_list = []

        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos/"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos/"+v+"/"
            images_list = sorted(os.listdir(video_folder))
            for i in range(0,len(images_list), 6):
                waymo_list = waymo_list + [video_folder+images_list[i], video_folder+images_list[i+1], video_folder+images_list[i+2]]

        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"+v+"/"
            images_list = sorted(os.listdir(video_folder))
            for i in range(0, len(images_list), 2):
                bdd100k_list.append(video_folder+images_list[i])

        for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"):
            video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"+v+"/"
            images_list = sorted(os.listdir(video_folder))
            for i in range(0, len(images_list), 3):
                argoverse_list.append(video_folder+images_list[i])

        self.images_list = waymo_list + bdd100k_list + argoverse_list
        print("Now the images are: {}".format(len(self.images_list)))

        for id in [1,2]:#self.ids_list[:10]:
            file_name = self.images_lookup_table[id]
            shutil.copy(file_name,"/content/drive/MyDrive/VISIOPE/Project/data/images")
            i = [i for i,c in enumerate(file_name[::-1]) if c=="/"][0]
            im = file_name[len(file_name)-i]
            os.rename("/content/drive/MyDrive/VISIOPE/Project/data/images/"+im, "/content/drive/MyDrive/VISIOPE/Project/data/images/"+id+".jpg")
        
        
    def extract_labels(self):
        coco_waymo = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json")
        coco_bdd100k = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json")
        coco_argoverse = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json")
        annotations = coco_waymo.dataset["annotations"] + coco_bdd100k.dataset["annotations"] + coco_argoverse.dataset["annotations"] 
        d = []
        for file_name in self.images_list[:10]: # dovrò essere in grado di accederci 
            value = list( map(lambda y: str(y["category_id"])+" "+str(y["bbox"][0]+" "+str(y["bbox"][1])+" "+str(y["bbox"][2]))+" "+str(y["bbox"][3]), list(filter(lambda x: x["file_name"]==file_name, annotations))) )
            d.append({file_name : value})
            
        for file_name in self.images_list[:10]:
            f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/"+self.images_lookup_table[file_name]+".txt", "w")
            f.write('\n'.join(d[file_name]))
            f.close
    
    def extract(self):
        if self.image_or_label == "image":
            self.extract_images()
        else:
            self.extract_labels()