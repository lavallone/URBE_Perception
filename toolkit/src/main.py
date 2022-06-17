import time
import os
import glob
import shutil
import threading
from datetime import timedelta
import WaymoOpenDataset

## Util functions ##
def clean_directory(files):
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

                
def remove_directory(save_dir, list_processed_segments):
    list_NOT_processed_segments = os. listdir(save_dir)[1:]
    ris = []
    for e in list_NOT_processed_segments:
        if e not in list_processed_segments:
           ris.append(e)
           
    for dir in ris:
        try:
            shutil.rmtree(dir)
        except OSError as e:
            print("Error: %s : %s" % (dir, e.strerror))


if __name__=="__main__":

    #### COLAB ####
    training_dir = "/content/drive/MyDrive/VISIOPE/Project/dataset/training" # provide directory where .tfrecords are stored
    save_dir = "/content/dataset/training" # provide a directory where data should be extracted
    
    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)
    
    # clear images and labels from previous executions
    clean_directory( glob.glob('{}/**/*.txt'.format(save_dir), recursive=True) )
    clean_directory( glob.glob('{}/**/*.json'.format(save_dir), recursive=True) )
    clean_directory( glob.glob('{}/**/*.png'.format(save_dir), recursive=True) )
    
    iteration = 0
    list_processed_segments = []
    for segment in toolkit.list_training_segments(): # mi creo una lista di segmenti...
        iteration = iteration + 1
        
        toolkit.assign_segment(segment)
        
        elapsed=0
        if  iteration != 1:
            start = time.time()
            list_processed_segments.append(segment)
            #t = threading.Thread(target=toolkit.extract_camera_images)
            #t.start()
            #t.join()
            end = time.time()
            elapsed = end - start
        
        #toolkit.save_video()
        #toolkit.consolidate()
        print(timedelta(seconds=elapsed))
        print(list_processed_segments)
        
        if iteration == 3: # for controlling how many segments we're going to process
            break 
    
    print("################# Processing is Finished ;) #################")
    print("Number of processed segments: {}".format(len(list_processed_segments)))
    print("removing the useless and empty directories...")
    #remove_directory(save_dir, list_processed_segments)
    print("Done!")