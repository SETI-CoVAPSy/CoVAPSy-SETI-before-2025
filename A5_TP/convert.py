from PIL import Image
import numpy as np

def to_jpg(image_path, output_path, dimensions=(256, 256)):
    with open(image_path, 'rb') as f:
        img = np.frombuffer(f.read(), dtype=np.float32)
        
    img = img.astype(np.uint8)
    img = img.reshape((3, dimensions[0], dimensions[1]))
    img = np.transpose(img, (2, 1, 0))
    img = Image.fromarray(img)
    #img = img.rotate(270)
    img.save(output_path)

def to_raw(image_path, output_path, dimensions=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(dimensions)
    img = np.array(img)
    img = np.transpose(img, (2, 1, 0))
    img = img.astype(np.float32)
    
    with open(output_path, 'wb') as f:
        f.write(img.tobytes())
        
if __name__ == '__main__':
    #to_jpg('ferrari.raw', 'test1.jpg', (1280, 960))
    # to_raw('Image/Bugatti.jpg', 'Image/base_image.raw', (4736, 2664))
    #to_jpg('tmp.raw', 'tmp.jpg', (1280, 960))
    
    to_jpg("Image/base_image_out_GPU_int.raw", "output/base_image_out_GPU_int.jpg", (256,144))
    to_jpg("Image/base_image_out_CPU.raw", "output/base_image_out_CPU.jpg", (256,144))
    to_jpg("Image/base_image_out_GPU.raw", "output/base_image_out_GPU_int.jpg", (256,144))
    to_jpg("Image/ferrari_out_GPU.raw", "output/ferrari_out_GPU.jpg", (256, 144))
    to_jpg("Image/ferrari_out_GPU_int.raw", "output/ferrari_out_GPU_int.jpg", (256, 144))
    

def resize(img, size, aspect_ratio=(16, 9)):
    
    # Get original dimensions
    width, height = img.size
    
    # Calculate the target dimensions based on the aspect ratio
    aspect_width, aspect_height = aspect_ratio
    target_height = int(width * aspect_height / aspect_width)
    
    # If the target height exceeds the original image height, adjust the width
    if target_height > height:
        target_width = int(height * aspect_width / aspect_height)
        target_height = height
    else:
        target_width = width
    
    # Calculate cropping box
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = img.resize(size)
    return resized_img