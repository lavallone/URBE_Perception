import time
import os
import argparse
import glob
import shutil
import threading
from datetime import timedelta
import WaymoOpenDataset

def process_segment():
    start = time.time()
    t = threading.Thread(target=toolkit.extract_camera_images)
    t.start()
    t.join()
    end = time.time()
    elapsed = end - start
    print(timedelta(seconds=elapsed))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("mode", type=str, default="add")
    #args = parser.parse_args()
    
    #### COLAB ####
    training_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/prova" # provide directory where .tfrecords are stored
    save_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/images/train/videos" # provide a directory where data should be extracted to
    
    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)
    
    
    iteration = 0
    list_segments = toolkit.list_training_segments()
    num_segments = len(list_segments)
        
    for segment in list_segments: # mi creo una lista di segmenti...
        iteration = iteration + 1
        num_segments = num_segments - 1
        print("^^^^^^^^^^^^^^^^^^^^^^ Starting processing |{}| ^^^^^^^^^^^^^^^^^^^^^^".format(segment[:-28]))
        if num_segments != 0:
            print("^^^^^^^^^^^^^^^^^^^^^^     {} segments left     ^^^^^^^^^^^^^^^^^^^^^^".format(num_segments))
        else:
            print("^^^^^^^^^^^^^^^^^^^^^^  Last segment to process ^^^^^^^^^^^^^^^^^^^^^^")
        toolkit.assign_segment(segment)
            
        process_segment()
        #toolkit.save_video()
        #toolkit.consolidate()
            
        if iteration == 100: # for controlling how many segments we're going to process
            break 
        
    print("################# Processing is Finished ;) #################")
    print("Number of processed segments: {}".format(iteration))