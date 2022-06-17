import time
import os
import threading
from datetime import timedelta
import WaymoOpenDataset

## Util functions ##
def clean_directory(root_path):
    for root, _, files in os.walk(root_path):
      for name in files:
        if name.endswith((".png")) or name.endswith((".json")) or name.endswith((".txt")):
            try: 
                f = os.path.join(root, name)
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

if __name__=="__main__":

    #### COLAB ####
    training_dir = "/content/drive/MyDrive/VISIOPE/Project/dataset/training" # provide directory where .tfrecords are stored
    save_dir = "/content/dataset/training" # provide a directory where data should be extracted
    
    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)
    
    # clear images and labels from previous executions
    #delete_files(glob.glob("{}/{}/*.png".format(camera_images_dir), recursive=True))
    clean_directory(save_dir)
    
    iteration = 0
    for segment in toolkit.list_training_segments(): # mi creo una lista di segmenti...
        iteration = iteration + 1
        
        toolkit.assign_segment(segment)
        
        start = time.time()
        t = threading.Thread(target=toolkit.extract_camera_images)
        t.start()
        t.join()
        end = time.time()
        elapsed = end - start
        
        #toolkit.save_video()
        #toolkit.consolidate()
        print(timedelta(seconds=elapsed))
        if iteration == 3:
            break 