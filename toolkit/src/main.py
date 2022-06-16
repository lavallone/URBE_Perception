import time
import threading
from datetime import timedelta
import WaymoOpenDataset

if __name__=="__main__":

    #### COLAB ####
    training_dir = "/content/drive/MyDrive/VISIOPE/Project/dataset/training/tfrecord" # provide directory where .tfrecords are stored
    save_dir = "/content/drive/MyDrive/VISIOPE/Project/dataset/training/processed" # provide a directory where data should be extracted

    toolkit = WaymoOpenDataset.ToolKit(training_dir=training_dir, save_dir=save_dir)

    for segment in toolkit.list_training_segments(): # mi creo una lista di segmenti...
        toolkit.assign_segment(segment)
        
        start = time.time()
        print("ao")
        toolkit.extract_camera_images
        print("porco dio ")
        end = time.time()
        elapsed = end - start
        
        #toolkit.save_video()
        #toolkit.consolidate()
        print(timedelta(seconds=elapsed))
        break # in questo modo processo un solo segmento