import os
import glob
import random
from PIL import Image
import json
from pycocotools.coco import COCO
from tqdm import tqdm

def uniqueid():
    seed = 0
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
    def __init__(self, img2id=None, img2oldID=None, oldID2id=None, images_list=None, old_ids_list=None):

        self.img2id = img2id
        self.img2oldID = img2oldID
        self.oldID2id = oldID2id
        self.images_list = (json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/images_list.json")))["images_list"]
        self.old_ids_list = old_ids_list # all'inizio è 'None'
        
        self.processed_images_so_far = {"images_so_far" : []}
        
    def extract_images(self):
        print("Starting extracting images...")
        
        # # first of all, we delete the previous images inside the folder
        # print("Deleting the previous images...")
        # for f in glob.glob('{}/*.jpg'.format("/content/drive/MyDrive/VISIOPE/Project/data/images"), recursive=True):
        #     os.remove(f)
        # print("Done!")
        
        ################################################################################
        #### IF WE ALREADY HAVE THE 'images_list.json' file WE DON'T NEED THIS CODE ####
        ################################################################################
        # we select images because many of them are similar (subsequent frame images)
        # waymo_list = []
        # bdd100k_list = []
        # argoverse_list = []
        
        # print("Processing Waymo images...")
        # for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos"):
        #     video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/videos/"+v
        #     images_list = sorted(os.listdir(video_folder))
        #     for i in range(0,len(images_list), 9):
        #         waymo_list = waymo_list + [video_folder+"/"+images_list[i], video_folder+"/"+images_list[i+1], video_folder+"/"+images_list[i+2]]
        
        # print("Processing BDD100K images...")
        # for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"):
        #     video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/images/videos/"+v+"/"
        #     images_list = sorted(os.listdir(video_folder))
        #     for i in range(0, len(images_list), 3):
        #         bdd100k_list.append(video_folder+images_list[i])
                
        # print("Processing Argoverse images...")
        # for v in os.listdir("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"):
        #     video_folder = "/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/images/videos/"+v+"/"
        #     images_list = sorted(os.listdir(video_folder))
        #     for i in range(0, len(images_list), 6):
        #         argoverse_list.append(video_folder+images_list[i])
                
        # self.images_list = waymo_list + bdd100k_list + argoverse_list
        # random.shuffle(self.images_list) # for shuffling the order of the images
        # print("Now the images are: {}".format(len(self.images_list)))
        # we save  the list for future purposes
        # f = open("/content/drive/MyDrive/VISIOPE/Project/data/images_list.json", "w")
        # d = {"images_list" : self.images_list}
        # json.dump(d, f)
        # f.close()
        ################################################################################
        
        # creo anche la lista dei vecchi IDs
        self.old_ids_list = []
        for img in self.images_list:
            self.old_ids_list.append(self.img2oldID[img])
        
        # self.processed_images_so_far = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/processed_images_so_far.json"))
        # step = self.processed_images_so_far["images_so_far"][-1][0]
        # self.images_list = self.images_list[step:]
        
        # print("Saving the new images to 'data/images'...")
        # image_list = []
        # for file_name in tqdm(self.images_list):
        #     step+= 1 
        #     id = self.img2id[file_name]
        #     name = name_id(id, 6)
        #     name += '.jpg'
            
        #     im = Image.open(file_name)
        #     resized_im = im.resize((1280, 720))
        #     final_im = resized_im.convert("RGB")
        #     image_list.append(final_im)
        #     self.processed_images_so_far["images_so_far"].append((step, file_name))
        #     if step%500 == 0: # actually save the images
        #       for f_i in image_list:
        #           f_i.save('/content/drive/MyDrive/VISIOPE/Project/data/images/'+ name)
        #       image_list = []
        #       f = open("/content/drive/MyDrive/VISIOPE/Project/data/processed_images_so_far.json", "w")
        #       json.dump(self.processed_images_so_far, f)
        #       f.close()
        # print("Done!")
        
    def extract_labels(self):
        print("Starting extracting labels...")
        coco_waymo = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/labels/COCO/annotations.json")
        coco_bdd100k = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/BDD100K/labels/COCO/annotations.json")
        coco_argoverse = COCO("/content/drive/MyDrive/VISIOPE/Project/datasets/Argoverse/labels/COCO/annotations.json")
        images = coco_waymo.dataset["images"] + coco_bdd100k.dataset["images"] + coco_argoverse.dataset["images"]
        annotations = coco_waymo.dataset["annotations"] + coco_bdd100k.dataset["annotations"] + coco_argoverse.dataset["annotations"] 
        
        # loading the new annotations file
        new_annotations = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/labels/COCO/annotations.json"))
        
        self.processed_images_so_far = json.load(open("/content/drive/MyDrive/VISIOPE/Project/data/processed_images_so_far.json"))
        
        step = self.processed_images_so_far["images_so_far"][-1][0]
        self.images_list = self.images_list[step:]
        self.old_ids_list = self.old_ids_list[step:]
        
        print("Create new annotations...")
        id_generator = uniqueid()
        new_images_list = list(filter(lambda x: x["id"] in self.old_ids_list, images))
        new_annotations_list = list(filter(lambda x: x["image_id"] in self.old_ids_list, annotations))
        print(len(new_images_list))
        l=[i["image_id"] for i in new_annotations_list]
        l=list(set(l))
        print(len(l))
        # for file_name,image_id in tqdm(zip(self.images_list, self.old_ids_list)):
        #     #--------------------------------------------------------------------------#
        #     step += 1
        #     d = {}
        #     im = list(filter(lambda x: x["id"]==self.img2oldID[file_name], new_images_list))[0] 
        #     new_images_list.remove(im) # rimuovo per efficienza futura computazionale      
        #     id = self.img2id[file_name]
        #     id = name_id(id, 6)
        #     d["id"] = id
        #     d["file_name"] = file_name
        #     d["width"] = 1280
        #     d["height"] = 720
        #     d["timeofday"] = im["timeofday"]
        #     new_annotations["images"].append(d)
        #     #--------------------------------------------------------------------------#
        #     annot = list(filter(lambda x: x["image_id"]==image_id, new_annotations_list))
        #     for ann in annot:
        #         new_image_id = self.img2id[file_name]
        #         new_image_id = name_id(new_image_id, 6)
        #         ann["image_id"] = new_image_id
        #         new_id = str(next(id_generator))
        #         new_id = name_id(new_id, 8)
        #         ann["id"] = new_id
        #         new_annotations["annotations"].append(ann)
        #         new_annotations_list.remove(ann) # per efficienza
        #     self.processed_images_so_far["images_so_far"].append((step, file_name))
        #     #--------------------------------------------------------------------------#
        #     if step%500 == 0: # save the processed images
        #         f = open("/content/drive/MyDrive/VISIOPE/Project/data/processed_images_so_far.json", "w")
        #         json.dump(self.processed_images_so_far, f)
        #         f.close()
        #         f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/COCO/annotations.json", "w")
        #         json.dump(new_annotations, f)
        #         f.close()
            
        # print("Done!")
        
        # # print("Saving the new annotations files to 'data/labels'...")
        # # for file_name,image_id in tqdm(zip(self.images_list, self.old_ids_list)):
        # #     annot = list(filter(lambda x: x["image_id"]==image_id, annotations))
        # #     for ann in annot:
        # #         new_id = self.img2id[file_name]
        # #         new_id = name_id(new_id, 6)
        # #         ann["image_id"] = new_id
        # #         new_annotations["annotations"].append(ann)
        # # print("Done!")
        
        # # number of annotations
        # print("Total number of annotations: " + str(len(new_annotations["annotations"])))
        
        # # # standardizziamo e unifichiamo gli ID delle annotazioni
        # # print("Setting annotations IDs to be unique!")
        # # id_generator = uniqueid()
        # # for ann in new_annotations["annotations"]:
        # #     id = str(next(id_generator))
        # #     id = name_id(id, 8)
        # #     ann["id"] = id
        # # print("Done!")
        
        # print("Writing the 'annotations.json' file...")
        # f = open("/content/drive/MyDrive/VISIOPE/Project/data/labels/COCO/annotations.json", "w")
        # json.dump(new_annotations, f)
        # f.close()
        # print("Done!")