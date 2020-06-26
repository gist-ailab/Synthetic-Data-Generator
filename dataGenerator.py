import os
from datetime import datetime
import numpy as np
from PIL import Image
from skimage import io
from os.path import join
import time
import sys
import argparse
import pickle

from environments.basic_env import BasicEnv
from externApi.fileApi import *

class Generator(object):

    def __init__(self, scene, max_img_num, ep_length=5, start_img_num=0, headless=False):
        # initialize environment
        self.env = BasicEnv(scene, headless=headless)

        # data generate setting        
        self.img_max = max_img_num
        self.img_num = start_img_num
        self.img_name = ""
        self.ep_length = ep_length
        self.ep_num = 0
        self.uint16_conversion = 10000 # detph value range 0.0001 ~ 10(m) => 1 mean 1mm

        self._initialize_dir()

    def _set_img_name(self):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.img_name = f"ep_{self.ep_num}_img_{self.img_num}_{timestamp}.png"

    def _is_ended(self):
        if self.img_num < self.img_max:
            return False
        else:
            return True
    
    def _initialize_dir(self):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.save_path = join(self.current_path, "image")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        #TODO: data save path
        self.rgb_dir = join(self.save_path, "rgb")
        check_and_create_dir(self.rgb_dir)
        self.depth_dir = join(self.save_path, "depth_value")
        check_and_create_dir(self.depth_dir)
        self.target_dir = join(self.save_path, "target_rgb")
        check_and_create_dir(self.target_dir)
        self.mask_dir = join(self.save_path, "mask")
        check_and_create_dir(self.mask_dir)
        
    def save_image(self):
        rgb_image, depth_raw, target_rgb, target_mask = self.env.get_images()
        self._set_img_name()
        
        np_name = self.img_name.replace(".png", ".npy") # save numpy 
        depth_value = np.uint16(depth_raw * self.uint16_conversion)
        rgb_image = Image.fromarray(np.uint8(rgb_image * 255))        
        target_image = Image.fromarray(np.uint8(target_rgb * 255))
        target_mask = Image.fromarray(target_mask)
        
        try:
            rgb_image.save(join(self.rgb_dir, self.img_name))
            target_image.save(join(self.target_dir, self.img_name))
            np.save(join(self.depth_dir, np_name), depth_value)
            target_mask.save(join(self.mask_dir, self.img_name))
            
        except FileNotFoundError:
            print("fail to save data")

    def run_episode(self):
        self.env.reset()
        for i in range(self.ep_length):
            self.env.step()
            self.save_image()
            print('[INFO] successfully saved image: ', self.img_name)
            self.img_num += 1

    def start(self):
        while not self._is_ended(): 
            self.run_episode()
            self.ep_num += 1            
        self.env.shutdown()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument("--scene", type=str, default="basic_scene.ttt", help="scene file")
    parser.add_argument("--pid", type=int, default=1, help="process id. 1 from 16")
    # object numbers
    parser.add_argument("--max_img_num", type=int, default=10000, help="maximum image number")
    parser.add_argument("--headless", action='store_true', help='no gui if true')
    args = parser.parse_args()

    # generate data
    generator = Generator(scene=args.scene,
                          max_img_num=args.max_img_num,
                          ep_length=20,
                          headless=args.headless,
                          ) 
    start_time1 = time.time()
    generator.start()
