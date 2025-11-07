import subprocess
import io
from threading import Thread
import cv2

class FfmpegVideo():
    
    def __init__(self, output_path, framerate=30):
        self.output_path = output_path
        self.closed = False
        self.ffmpeg_proc = subprocess.Popen(f"ffmpeg -y -stats -framerate {framerate} -f image2pipe -c:v mjpeg -i - -c:v copy -r {framerate} {output_path}", shell=True, stdin=subprocess.PIPE)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def add_image(self, data):
        if self.closed: return
        
        try:
            self.ffmpeg_proc.stdin.write(data)
        except (ValueError, BrokenPipeError):
            print(f"Broken pipe {self.output_path}")
            self.closed = True
            
    def add_pil_image(self, pil_image):
        with io.BytesIO() as output:
            pil_image.save(output, format="JPEG")
            self.add_image(output.getvalue())
            
    def add_opencv_image(self, opencv_image):
        print("Added image")
        _, buffer = cv2.imencode(".jpg", opencv_image)
        self.add_image(buffer)
        
    def stop(self):
        if not self.closed:
            self.ffmpeg_proc.stdin.close()
            self.ffmpeg_proc.wait()
            self.closed = True