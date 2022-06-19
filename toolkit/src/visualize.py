import cv2

# Add labels and bounding boxes to verify the data
def process_image(image, labels):
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    for label in labels:
        print(label)
        label_list = list(map(float, label.split(" ")))
        startPoint = (int(label_list[1]), int(label_list[2]))
        sizePoint = (int(label_list[1] + label_list[3]), int(label_list[2] + label_list[4]))
        image = cv2.rectangle(image, startPoint, sizePoint, color=(255, 0, 0), thickness=3)
    return image

if __name__ == "__main__":
    
    # visualize images
    saveDir = "../data" # directory where the data was extracted
    imageDir = "{}/camera/images".format(saveDir)
    labelDir = "{}/camera/labels".format(saveDir)
    frameNum = 0
    camera_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    
    for camera in camera_list:
        img = cv2.imread("{}/{}_{}.png".format(imageDir, frameNum, camera), cv2.IMREAD_UNCHANGED)
        label = open("{}/{}_{}.txt".format(labelDir, frameNum, camera), "r")
        cv2.imshow("{}_{}".format(frameNum, camera), img)
    cv2.waitKey()
    cv2.destroyAllWindows()