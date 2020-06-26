import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import io
from skimage.color import rgb2gray, gray2rgb
from PIL import Image
from datetime import datetime


current_path = os.path.dirname(os.path.realpath(__file__))

def imageCrawling(keyword, dir):
    from google_images_download import google_images_download
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    response = google_images_download.googleimagesdownload()

    arguments = {"keywords":keyword,
                 "limit":100,
                 "print_urls":True,
                 "no_directory":True,
                 "output_directory":dir}

    paths = response.download(arguments)
    print(paths)

def get_image_files(path):
    image_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    
    return image_files

def image2gray2rgb(image_file):
    try:
        img = io.imread(image_file)
        # img_gray = rgb2gray(img)
    
        # noise = (np.random.rand() + 1) / 2
        # img_gray = (img_gray * noise) 
        img_rgb = gray2rgb(img)
        # io.imsave(image_file, img_gray)
        io.imsave(image_file, img_rgb)
    except:
        os.remove(image_file)

def get_random_grad_img(image, first_color, second_color):
    """get random color gradient image
    
    Arguments:
        image {numpy array} -- [description]
    """
    w, h = image.size
    x = np.linspace(first_color, second_color, h)
    x = np.expand_dims(x, axis=1)
    y = np.linspace(first_color, second_color, w)
    
    grad_x = np.tile(x, (1, w, 1))
    grad_y = np.tile(y, (h, 1, 1))
    grad_img = np.uint8(255 * (grad_x + grad_y) / 2)
    # try:
    #     grad_img = Image.fromarray(np.uint8(grad_img*255)) 
    # except MemoryError:
    #     print("[ERROR] Memory Error occur: image size {}".format(image.size))
    #     grad_img = None
    return grad_img
    
def create_randomize_image(image_path, first_color, second_color, process_id):
    """from texture image to randomize color image
    
    Arguments:
        image_path {[type]} -- [texture image path]
        first_color {[type]} -- [description]
        second_color {[type]} -- [description]
    
    Returns:
        img_path [string] -- [randomized image path]
    """
    image = Image.open(image_path)
    
    rand_grad = get_random_grad_img(image, first_color, second_color)
    image = np.array(image)
    if not type(rand_grad) == np.ndarray:
        return None 
    else:
        try:
            # rand_img = Image.blend(image, rand_grad, alpha = 0.5)
            rand_img = (image[:,:,:3] / 2 + rand_grad / 2)
        except:
            print(image.size)
            return None
        # now = datetime.now()
    # timestamp = datetime.timestamp(now)
    rand_img = Image.fromarray(np.uint8(rand_img))
    img_path = current_path + "/randomize_textures/texture" +str(process_id) + ".png"
    try:
        rand_img.save(img_path)
    except:
        os.mkdir(os.path.join(current_path, "randomize_textures"))
        rand_img.save(img_path)
    
    return img_path

def set_texture_grad(image_path, first_color, second_color):
    image = Image.open(image_path)
    rand_grad = get_random_grad_img(image, first_color, second_color)
    image = np.array(image)
    
    if not type(rand_grad) == np.ndarray:
        return None 
    else:
        # try:
        #     rand_img = Image.blend(image, rand_grad, alpha = 0.5)
        # except :
        #     print(image.size)
        #     return None
        # rand_img = Image.blend(image, rand_grad, alpha = 0.5)
        rand_img = (image[:,:,:3] / 2 + rand_grad / 2)
        # now = datetime.now()
    # timestamp = datetime.timestamp(now)
    rand_img = Image.fromarray(np.uint8(rand_img))
    img_path = image_path.replace("_original", "")
    try:
        rand_img.save(img_path)
    except:
        print("ERROR")

def get_segmentation_image(color, process_id):
    """make segmentation color image
    
    Arguments:
        color {list(255)} -- [RGB]
    
    Returns:
        [type] -- [description]
    """
    image_path = current_path + "/segmentation_textures/segmentation" +str(process_id) +".png"
    image = Image.new("RGB", (256,256), color)
    try:
        image.save(image_path)
    except:
        os.mkdir(os.path.join(current_path, "segmentation_textures"))
        image.save(image_path)
    
    return image_path

# path = current_path + "/textures/"
# # # imageCrawling('texture', path)
# image_files = get_image_files(path)

# for image_path in image_files:
#     create_randomize_image(image_path, [1,0,0],[0,1,0])
# # #     image2gray2rgb(image_path)
