import os
import threading
import json
import random

# function for generating unique ids
#####################################
def uniqueid():
    seed = random.getrandbits(20)
    while True:
       yield seed
       seed += 1
#####################################

class BDD100KToolKit:
    def __init__(self, labels_json=None, labels_dir=None):

        self.get_id = uniqueid()
        
        self.labels_dir = labels_dir
        self.labels_json = labels_json
        
        self.json_dictionary = json.load(open(labels_json))
        
    def list_json_videos(self):
        l = []
        for file in os.listdir(self.labels_dir):
            if file.endswith(".json"):
                l.append(file)
        return l
        
    def extract_labels(self, json_video):
        
            d = json.load(open(self.labels_dir+"/"+json_video))
            name_video = d[0]["videoName"]
            totalFrames = d[-1]["frameIndex"] + 1
            self.update_json_video(name_video, totalFrames)

            list_image = []
            for image_dict in d:
                name_image = name_video +"/" + image_dict["name"]
                #id_image = name_video + name_image[:-4]
                image_id = next(self.get_id)
                width = 1280
                height = 720
                list_image.append({"id" : image_id, "file_name" : name_image, "video_id" : name_video, "width" : width, "height" : height})
                list_labels = []
                for label in image_dict["labels"]:
                    if label["category"] == "car" or label["category"] == "truck" or label["category"] == "bus" or label["category"] == "other vehicle" or label["category"] == "pedestrian" or label["category"] == "rider" or label["category"] == "other person" or label["category"] == "bicycle" or label["category"] == "motorcycle":
                        if label["attributes"]["truncated"] == False and label["attributes"]["crowd"] == False:
                            id = label["id"]
                            # Ipotizzando che (x1,y1) è l'angolo sx di sotto e (x2,y2) quello dx di sopra...
                            x1 = label["box2d"]["x1"]
                            y1 = label["box2d"]["y1"]
                            x2 = label["box2d"]["x2"]
                            y2 = label["box2d"]["y2"]
                            w = x2-x1
                            h = y2-y1
                            bbox = [x1, y1, w, h]
                            if label["category"] == "car" or label["category"] == "truck" or label["category"] == "bus" or label["category"] == "other vehicle":
                                cat_id = 0
                            elif label["category"] == "pedestrian" or label["category"] == "rider" or label["category"] == "other person":
                                cat_id = 1
                            else:
                                cat_id = 2
                            list_labels.append({"id" : id, "image_id" : image_id, "category_id" : cat_id, "bbox" :  bbox})
                self.update_json_annotation(list_labels)
            self.update_json_image(list_image)
           
        
    def bdd100k_extraction(self): # provo a implementarlo col multi-threading
        
        iteration = 0
        list_json_videos = self.list_json_videos()
        num_json_video = len(list_json_videos)
            
        for json_video in list_json_videos: # 1200 videos
            iteration = iteration + 1
            num_json_video = num_json_video - 1
            print("^^^^^^^^^^^^^^^^^^^^^^ Starting processing {} ^^^^^^^^^^^^^^^^^^^^^^".format(json_video))
            if num_json_video != 0:
                print("^^^^^^^^^^^^^^^^^^^^^^     {} json files left     ^^^^^^^^^^^^^^^^^^^^^^".format(num_json_video))
            else:
                print("^^^^^^^^^^^^^^^^^^^^^^  Last json file to process ^^^^^^^^^^^^^^^^^^^^^^")
            
            # appena inizio a processare un video, carico il dizionario AGGIORNATO dal json file
            #self.json_dictionary = json.load(open(self.labels_json))
            t = threading.Thread(target=self.extract_labels, args=[json_video])
            t.start()
            t.join()
            
            # appena finisco vado a salvare le modifiche apportate e le salvo sullo stesso json file
            #json.dump(self.json_dictionary, open(self.labels_json, "w"))
            
            # if iteration % 100 == 0: # ogni 100 video, per alleggerire il carico, inizio a salvare il 'self.json_dictionary' corrente.
            #     f = open(self.labels_json, "w")
            #     json.dump(self.json_dictionary, f)
            #     f.close()
                
            if iteration == 200#10000:
                break
            
        print("################# Processing is Finished ;) #################")
        print("Number of processed json files: {}".format(iteration))
        print("loading the new label_json file...")
        f = open(self.labels_json, "w")
        json.dump(self.json_dictionary, f) 
        print("Done!")    
            
    def update_json_video(self, name, num_frames, time_of_day=None):
        self.json_dictionary["videos"].append({"id" : name, "num_frames" : num_frames, "time" : time_of_day})
        
    def update_json_image(self, list):
        self.json_dictionary["images"] = self.json_dictionary["images"] + list
        
    def update_json_annotation(self, list):
        self.json_dictionary["annotations"] = self.json_dictionary["annotations"] + list