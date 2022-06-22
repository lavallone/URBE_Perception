import os
import threading
import json
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

class BDD100KToolKit:
    def __init__(self, labels_json=None, labels_dir=None):

        self.json_video = None
        
        self.labels_dir = labels_dir
        self.labels_json = labels_json
        
        self.json_dictionary = json.load(open(labels_json))
        
    def list_json_videos(self):
        l = []
        for file in os.listdir(self.labels_dir):
            if file.endswith(".json"):
                l.append(file)
        return l
        
    def extract_labels(self):
        
        d = json.load(open(self.json_video))
        name_video = d[0]["videoName"]
        totalFrames = d[-1]["frameIndex"] + 1
        self.update_json_video(name_video, totalFrames)

        list_image = []
        for image_dict in d:
            name_image = image_dict["name"]
            id_image = name_video + name_image[:-5]
            width = 1280
            height = 720
            list_image.append({"id" : id_image, "name" : name_image, "video_id" : name_video, "width" : width, "height" : height})
            list_labels = []
            for label in image_dict["labels"]:
                if label["category"] == "car" or label["category"] == "pedestrian" or label["category"] == "bycicle":
                    if label["attributes"]["occluded"] == False and label["attributes"]["truncated"] == False:
                        id = label["id"]
                        # Ipotizzando che (x1,y1) è l'angolo sx di sopra e (x2,y2) quello dx di sotto...
                        x1 = label["bbox"]["x1"]
                        y1 = label["bbox"]["y1"]
                        x2 = label["bbox"]["x2"]
                        y2 = label["bbox"]["y2"]
                        w = x2-x1
                        h = y1-y2
                        bbox = [x1, x2, w, h]
                        if label["category"] == "car":
                            cat_id = 0
                        elif label["category"] == "pedestrian":
                            cat_id = 1
                        else:
                            cat_id = 2
                        list_labels.append({"id" : id, "image_id" : id_image, "category_id" : cat_id, "bbox" :  bbox})
            self.update_json_annotation(list_labels)
        self.update_json_image(list_image)
           
        
    def bdd100k_extraction(self):
        
        iteration = 0
        list_json_videos = self.list_json_videos()
        num_json_video = len(list_json_videos)
            
        for json_video in list_json_videos:
            iteration = iteration + 1
            num_json_video = num_json_video - 1
            print("^^^^^^^^^^^^^^^^^^^^^^ Starting processing |{}| ^^^^^^^^^^^^^^^^^^^^^^".format(json_video))
            if num_json_video != 0:
                print("^^^^^^^^^^^^^^^^^^^^^^     {} json files left     ^^^^^^^^^^^^^^^^^^^^^^".format(num_json_video))
            else:
                print("^^^^^^^^^^^^^^^^^^^^^^  Last json file to process ^^^^^^^^^^^^^^^^^^^^^^")
            self.json_video = json_video
            
            t = threading.Thread(target=self.extract_labels)
            t.start()
            t.join()
                
            if iteration == 10000: # for controlling how many segments we're going to process
                break 
            
        print("################# Processing is Finished ;) #################")
        print("Number of processed json filess: {}".format(iteration))
        print("loading the new label_json file...")
        f = open(self.labels_json, "w")
        json.dump(self.json_dictionary, f) 
        print("Done!")
        
            
    def update_json_video(self, name, num_frames, time_of_day=None, weather=None):
        d = self.json_dictionary
        d["videos"].append({"id" : name, "num_frames" : num_frames, "time" : time_of_day, "weather" :weather })
        self.json_dictionary = d
        
    def update_json_image(self, list):
        d = self.json_dictionary
        d["images"] = d["images"] + list
        self.json_dictionary = d
        
    def update_json_annotation(self, list):
        d = self.json_dictionary
        d["annotations"] = d["annotations"] + list
        self.json_dictionary = d