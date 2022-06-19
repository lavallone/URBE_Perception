import time
import os
import argparse
import glob
import shutil
import threading
from datetime import timedelta
import WaymoOpenDataset

def remove_directory(save_dir, list_processed_segments):
    list_NOT_processed_segments = os.listdir(save_dir)[1:]
    ris = []
    for i in range(len(list_NOT_processed_segments)):
        e = list_NOT_processed_segments[i]
        e = e + "_with_camera_labels.tfrecord"
        if e not in list_processed_segments:
           ris.append(e[:-28])
                      
    for dir in ris:
        if dir == "last_file.txt":
            continue
        try:
            shutil.rmtree(save_dir + "/" + dir)
        except OSError as e:
            print("Error: %s : %s" % (dir, e.strerror))

def process_segment():
    start = time.time()
    list_processed_segments.append(segment)
    t = threading.Thread(target=toolkit.extract_camera_images)
    t.start()
    t.join()
    end = time.time()
    elapsed = end - start
    print(timedelta(seconds=elapsed))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="add")
    args = parser.parse_args()
    
    #### COLAB ####
    training_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/training/archived_files" # provide directory where .tfrecords are stored
    save_dir = "/content/drive/MyDrive/VISIOPE/Project/datasets/Waymo/training/individual_files" # provide a directory where data should be extracted to
    
    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)
    
    if args.mode == "add":
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
        print("Number of processed segments: {}".format(num_segments))
        
    elif args.mode == "del": # abbiamo intenzione di eliminare la cartelle relative a questi segmenti
        print("removing the indicated segments directories...")
        print("[The ones saved in 'archived_files']")
        list_saved_segments = os.listdir(save_dir)[1:]
        list_to_delete_segments = toolkit.list_training_segments()
        
        for dir in list_to_delete_segments:
            if dir in list_saved_segments:
                try:
                    shutil.rmtree(save_dir + "/" + dir)
                except OSError as e:
                    print("Error: %s : %s" % (dir, e.strerror))
        print("Done!")