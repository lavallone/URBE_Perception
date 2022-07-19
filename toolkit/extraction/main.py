import argparse
from extract import ExtractionToolkit

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_lookup_table", type=dict)
    #parser.add_argument("image_or_label", type=str)
    args = parser.parse_args()
    
    toolkit = ExtractionToolkit(images_lookup_table=args.images_lookup_table)
    toolkit.extract_images()
    #toolkit.extract_labels()