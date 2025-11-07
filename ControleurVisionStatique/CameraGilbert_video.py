import numpy as np
import time
import cv2

class CameraGilbert():
    
    def __init__(self, *args, **kwargs):
        self.filename = "video.mkv"
        self.cap = None
    
    def enable(self, fps):
        self.cap = cv2.VideoCapture(self.filename)
        
    def getHeight(self):
        return self.cam.video_configuration.size[1]
        
    def getWidth(self):
        return self.cam.video_configuration.size[0]
    
    def getFov(self):
        return 62.2
        
    def getImage(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.enable(1)
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame from video")
        
        output = frame[:,:,::-1]
        
        return output

if __name__ == "__main__":
    camera = CameraGilbert()
    camera.enable(30)
    while True:
        image = camera.getImage()
        #cv2.imwrite("image.jpg", image)
        input("next ?")
        time.sleep(0.1)