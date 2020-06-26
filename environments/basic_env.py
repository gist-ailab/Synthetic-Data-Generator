from pyrep import PyRep
from pyrep.objects.shape import Shape

import os

from sensor.camera_manager import CameraManager
from sensor.light import Light

class BasicEnv(object):
    
    def __init__(self, scene_file=None, headless=False):
        """
        scene_file (*.ttt)
        """
        self.pr = PyRep()

        self._initialize_scene(scene_file, headless=headless)
        # get scene objects
        self.workspace = Shape("workspace")
        self.table = Shape("Table")
        self.lights = [Light("Light1"), Light("Light2"),
                       Light("Light3"), Light("Light4")]
        self.floors = [Shape("Floor1"), Shape("Floor2"), Shape("Floor3"), Shape("Floor4"), Shape("Floor5")]
        self.scene_objects = []

        # camera
        self.resolution = [640, 480]
        self.camera_manager = CameraManager(self.resolution)

        # labeling
        self.labeling_manager = LabelingManager(class_num=self.calss_num)

    def _initialize_scene(self, scene_file, headless):
        if type(scene_file) == str:
            scene_file = os.path.join(SCENE_DIR, scene_file)
            self.pr.launch(scene_file=scene_file, headless=headless)
        else:
            self.pr.launch(headless=headless)
        
        self.pr.start()

    def reset(self):
        self.camera_manager.reset()
        self.labeling_manager.reset()
        for scene_obj in self.scene_objects:
            scene_obj.remove()


    def step(self):
        self.pr.step()
    
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_images(self):
        self.camera_manager.capture()
            
        main_rgb = self.camera_manager.get_main_rgb()
        main_depth = self.camera_manager.get_main_depth()
        target_rgb = self.camera_manager.get_target_rgb()
        target_mask = self.camera_manager.get_target_mask()
        
        return main_rgb, main_depth, target_rgb, target_mask

class SceneObject(object):
    def __init__(self, scene_obj, is_dynamic, class_id):
        self.visible = scene_obj
        self.is_dynamic = is_dynamic
        
        if self.is_dynamic:
            self.respondable = self._get_dynamic_part()
            self.visible.set_parent(self.respondable)
        else:
            self.respondable = self.visible
            self.visible.set_parent(self.respondable)

        self.class_id = class_id

    def _get_dynamic_part(self):
        try:
            respondable = self.visible.get_convex_decomposition()
        except:
            respondable = self.visible.copy()
        respondable.set_renderable(False)
        respondable.set_dynamic(True)
        return respondable

    def set_position(self, position, relative_to=None):
        self.respondable.set_position(position, relative_to)
    def get_position(self, relative_to=None):
        return self.respondable.get_position(relative_to)
    
    def set_orientation(self, orientation, relative_to=None):
        self.respondable.set_orientation(orientation, relative_to)
    def get_orientation(self, relative_to=None):
        return self.respondable.get_orientation(relative_to)
    
    def set_pose(self, pose, relative_to=None):
        self.respondable.set_pose(pose, relative_to)
    def get_pose(self, relative_to=None):
        self.respondable.get_pose(relative_to)

    def remove(self):
        self.respondable.set_model(True)
        self.respondable.remove()

    @staticmethod
    def _create_by_CAD(filepath, scaling_factor, class_id, is_texture=False, is_dynamic=True):
        if is_texture:
            try:
                obj = Shape.import_shape(filename=filepath,
                                         scaling_factor=scaling_factor,)          
            except:
                print("error")
        else:
            try:
                obj = Shape.import_mesh(filename=filepath,
                                        scaling_factor=scaling_factor,)
            except:
                print("error")
        
        return SceneObject(obj, is_dynamic, class_id)

class LabelingManager(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.color_range = 255 / self.class_num
        self.class_colors = []
        for class_id in range(self.class_num): # 5
            rgb_min = [self.color_range * class_id + 1, self.color_range * class_id + 1 , self.color_range * class_id + 1] 
            rgb_max = [self.color_range * (class_id + 1), self.color_range * (class_id + 1), self.color_range * (class_id + 1)]
            self.class_colors.append((class_id, (rgb_min, rgb_max)))
        self.used_colors = []

    def get_class_color(self, class_id):
        color_range = self.class_colors[class_id]
        rgb_low = color_range[0]
        rgb_max = color_range[1]
        seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max)) # 0 ~ 255
        seg_color = tuple(seg_color)
        while seg_color in self.used_colors:
            seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max))
            seg_color = tuple(seg_color)
        self.used_colors.append(seg_color)

        return seg_color
    
    def reset(self):
        self.used_colors = []