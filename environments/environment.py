from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import TextureMappingMode

from os import listdir
import os
from os.path import isfile, join, isdir
import random
from colorsys import hsv_to_rgb
import datetime
import numpy as np

from sensor.azure import Azure
from sensor.light import Light
from sensor.camera_control import camera_control
from robot.panda import MyPanda

from externApi.imageProcessApi import create_randomize_image, get_segmentation_image, set_texture_grad
from externApi.fileApi import get_dir_list, get_file_list


class DREnv(object):

    def __init__(self, env_tag=0, headless=False, process_id=0, heavy_occlusion=False):
        """[summary]
        the environment for generate instance segmentation dataset
        objects: furniture(from cad file) (unknown/known)  
        Argument:
            env_tag {int} -- [0: gist env | 1: SNU env]
            headless {bool} -- [False: show screen | True: no screen]
            process_id {int} -- [image name index for preventing file collision]
        domain randomize: 
            1. table: randomize slightly
            2. floor & walls: randomize variously 
            3. objects: realistic(slightly color randomize)
            4. camera: various randomize
            5. light: 1 + 0~4
        segmentation rule:
            1. class range for semantic labeling
            2. instance color in class range
        """
        # randomization config.
        self._max_n_light = 4
        self._color_variance = 0.2
        self._texture_uv = [1, 1]
        self._min_z, self._max_z = 0.4, 2 #TODO: camera distance range
        self._z_rot_range = [-np.pi/4, np.pi/4] # -3 ~ 3 (in radians) z_rotation
        self._elevation_range = [0, (7 * np.pi / 18)] # 90 ~ 30 (in radians) y_rotation
        self._azimuth_range = [-np.pi/3, np.pi/3] # -60 ~ 60 (in radians) z_rotation
        self._min_pangle, self._max_pangle = 87, 93
        self._sim_stable_step = 5 # waiting for step * 0.05sec to stable state

        #------------------------------------------------------------------------ 

        self.is_heavy_occlusion = heavy_occlusion
        self.workspace_range = [(-0.2, -0.4), (0.8, 0.4)] # (x, y) value based on self.workspace
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.env_tag = env_tag
        self.process_id = process_id

        self.pr = PyRep()
        self._create_scene(headless=headless)
        # self.base_objects = self._create_base_objects()
        # self.base_objects = self._load_base_objects()
        self.base_objects = None

        """self.base_objects structure
        
        self.base_objects = (furnitures, connectors)
            furnitures = [
                furniture_part_1_info,
                furniture_part_2_info,
                ...
            ]
            connectors = [
                connector_part_1_info,
                ...
            ]
                each info = (
                    object_in_scene,
                    class_index,
                    z_max,(shortest axis length)
                    rotation_axis (shortest axis) 0: x, 1: y, 2: z
                )
        """
        self.base_path = self._get_base_objects_path()
        self.scene_furnitures = []
        self.scene_connectors = []
        """scene object info structure
        self.scene_furnitures = [
            copied_furniture_part_1_info,
            ...
        ]
            furniture_info = (
                respondable object,
                visible object,
                segmented object,
                segmentation color,
                object rotation axis
            )
        """

        # texture setting
        self.gray_textures = self._get_gray_texture_files()
        self.hard_textures = self._get_hard_texture_files()
        self.texture_shapes = []
        self.segmentation_textures = []

        # get scene objects
        self.workspace = Shape("workspace")
        self.invisible_walls = [Shape("invisible_wall0"), Shape("invisible_wall1"), 
                                Shape("invisible_wall2"), Shape("invisible_wall3")]
        self.table = Shape("Table")
        self.table_color = (0, 0, 255)
        self.seg_table = self._create_segmented_object(self.table, self.table_color)
        self.seg_table.set_parent(self.table)
        self.seg_table.set_renderable(False)
        self.lights = [Light("Light1"), Light("Light2"),
                       Light("Light3"), Light("Light4")]
        self.floors = [Shape("Floor1"), Shape("Floor2"), Shape("Floor3"), Shape("Floor4"), Shape("Floor5")]
        
        self._initialize_robots()
        # camera settings
        self.main_camera = Azure(Dummy("main_camera"))
        self.target_camera = Azure(Dummy("target_camera"))
        self.temp_camera = Azure(Dummy("temp_camera"))
        self.camera_rot_base = Dummy("camera_rot_base")
        self.target_base = Dummy('target')
        self.temp_base = Dummy("temp")
        self.camera_control = camera_control()
        self._initial_camera_pose = self.camera_control.get_pose(relative_to=self.camera_rot_base)
        
        self.pr.step()
        
    #region initialize scene
    def _create_scene(self, headless):
        """[launch the scene file and start simulation(coppeliasim)]
        
        Arguments:
            headless {bool} -- [start simulation without rendering]
        """
        if self.env_tag == 0:
            scene_file = self.current_path + "/scene/assembly_env_GIST.ttt"
        else:
            scene_file = self.current_path + "/scene/assembly_env_SNU.ttt" 
        # coppeliasim start
        self.pr.launch(scene_file=scene_file, headless=headless)
        self.pr.start()
        self.pr.step()

    def _create_base_objects(self):
        """load all obj files(stl/obj files)
        Arguments:
        
        Returns:
            furnitures, connectors {list} -- [loaded objects(same # as obj files)]
        """
        self.furniture_scaling_factor = 1
        self.connector_scaling_factor = 0.001
        # file path
        
        part_path = join(self.current_path, "part") 
        furnitures_path = join(part_path, "furniture_part")
        connectors_path = join(part_path, "connector_part")
        furniture_f_list = get_file_list(furnitures_path)
        connector_f_list = get_file_list(connectors_path)
        furniture_list = []
        connector_list = []
        furniture_names = []
        connector_names = []
        for file_path in furniture_f_list:
            part_name, file_ext = os.path.splitext(file_path)
            if 'obj' in file_ext:
                furniture_names.append(part_name.replace(furnitures_path, ""))
                furniture_list.append(file_path)
            else:
                pass

        for file_path in connector_f_list:
            part_name, file_ext = os.path.splitext(file_path)
            if 'obj' in file_ext:
                connector_names.append(part_name.replace(furnitures_path, ""))
                connector_list.append(file_path)
            else:
                pass
        furniture_list.sort()
        connector_list.sort()
        # segmentation range for class_id
        self.class_num = len(connector_list) + len(furniture_list) # objects to segmentation
        self.color_range = 255 / self.class_num
        self.class_colors = []
        for class_id in range(self.class_num): # 5
            rgb_min = [self.color_range * class_id + 1, self.color_range * class_id + 1 , self.color_range * class_id + 1] 
            rgb_max = [self.color_range * (class_id + 1), self.color_range * (class_id + 1), self.color_range * (class_id + 1)]
            self.class_colors.append((class_id, (rgb_min, rgb_max)))

        # import furnitures: 
        furnitures = []
        for furniture_id, obj in enumerate(furniture_list):
            try:
                shape = Shape.import_mesh(filename=obj,
                                          scaling_factor=self.furniture_scaling_factor)
            except:
                print("error occur: " + obj)
                continue
            name = "furniture_" + str(furniture_id)
            shape.set_name(name)
            shape.set_position([0, 0, -1])
            shape.set_renderable(False)
            z_max, shape_rotation_axis = self._get_rotation_axis(shape)
            class_id = furniture_id # segmentation id
            furnitures.append((shape, class_id, z_max, shape_rotation_axis))
        
        # import connectors:
        connectors = []
        for connector_id, obj in enumerate(connector_list):
            try:
                shape = Shape.import_mesh(filename=obj,
                                          scaling_factor=self.connector_scaling_factor)
            except:
                print("error occur: " + obj)
                continue
            name = "class_" + str(connector_id)
            shape.set_name(name)
            shape.set_position([0, 0, -1])
            shape.set_renderable(False)
            z_max, shape_rotation_axis = self._get_rotation_axis(shape)
            class_id = connector_id + len(furniture_list)
            connectors.append((shape, class_id, z_max ,shape_rotation_axis))

        print("[INFO] successfully import {} furnitures".format(len(furnitures)))
        print("[INFO] successfully import {} connectors".format(len(connectors)))
        print("[INFO] segmentation color setting is: \n")
        for i, fn in enumerate(furniture_names):
            print("\t Furniture: {} Class_ID: {} ".format(fn, furnitures[i][1]))
        for i, fn in enumerate(connector_names):
            print("\t Connector: {} Class_ID: {} ".format(fn, connectors[i][1]))
        for i, color_range in self.class_colors:
            print("\t Class_ID: {} object has {} ~ {} color range".format(i, color_range[0], color_range[1])) 
        print("\n")

        return (furnitures, connectors)
    
    def _load_base_objects(self):
        furniture_dummy = Dummy("furniture_base")
        connector_dummy = Dummy("connector_base")
        furniture_list = furniture_dummy.get_objects_in_tree()
        connector_list = connector_dummy.get_objects_in_tree()

        # segmentation range for class_id #TODO:
        self.class_num = len(connector_list) + len(furniture_list) # objects to segmentation
        self.color_range = 255 / self.class_num
        self.class_colors = []
        for class_id in range(self.class_num): # 5
            rgb_min = [self.color_range * class_id + 1, self.color_range * class_id + 1 , 0] 
            rgb_max = [self.color_range * (class_id + 1), self.color_range * (class_id + 1), 0]
            self.class_colors.append((class_id, (rgb_min, rgb_max)))

        furnitures = []
        furniture_names = []
        for furniture_id, shape in enumerate(furniture_list):
            furniture_name = shape.get_name()
            furniture_names.append(furniture_name)
            z_max, shape_rotation_axis = self._get_rotation_axis(shape)
            class_id = furniture_id # segmentation id #TODO:
            furnitures.append((shape, class_id, z_max, shape_rotation_axis))
        
        connectors = []
        connector_names = []
        for connector_id, shape in enumerate(connector_list):
            connector_name = shape.get_name()
            connector_names.append(connector_name)
            z_max, shape_rotation_axis = self._get_rotation_axis(shape)
            class_id = connector_id + len(furniture_list) #TODO:
            connectors.append((shape, class_id, z_max ,shape_rotation_axis))
        
        print("[INFO] successfully load {} furnitures in scene".format(len(furnitures)))
        print("[INFO] successfully load {} connectors in scene".format(len(connectors)))
        print("[INFO] segmentation color setting is: \n")
        for i, fn in enumerate(furniture_names):
            print("\t Furniture: {} Class_ID: {} ".format(fn, furnitures[i][1]))
        for i, fn in enumerate(connector_names):
            print("\t Connector: {} Class_ID: {} ".format(fn, connectors[i][1]))
        for i, color_range in self.class_colors:
            print("\t Class_ID: {} object has {} ~ {} color range".format(i, color_range[0], color_range[1])) 
        print("\n")

        return (furnitures, connectors)

    def _create_copied_object(self, cn_tag=-1):
        """[copy one objects in base objects]
        Arguments:
            cn_tag {int} -- [-1(default): furniture| 0~: connector_idx]
        """
        if cn_tag == -1:
            print("[INFO] copy furniture / ", datetime.datetime.now())
            picked_obj, class_idx, z_max, obj_rot_axis = random.choice(self.base_objects[0])
        else:
            print("[INFO] copy connector / ", datetime.datetime.now())
            picked_obj, class_idx, z_max, obj_rot_axis = self.base_objects[1][cn_tag]
        cop_obj_visible = picked_obj.copy()
        try:
            cop_obj = cop_obj_visible.get_convex_decomposition()
        except:
            cop_obj = cop_obj_visible.copy()
        cop_obj_visible.set_parent(cop_obj)
        cop_obj_visible.set_renderable(True)
        cop_obj.set_parent(self.target_base)
        cop_obj.set_renderable(False)
        self._randomize_object_texture(cop_obj_visible)
        # select segmentation color for object and create segmentation texture
        if not class_idx == -1: # -1 => no segmentation 
            class_color = self.class_colors[class_idx]
            obj_idx = class_color[0]
            if not class_idx == obj_idx:
                print("[ERROR] class index is not matching to class color") 
            color_range = class_color[1]
            rgb_low = color_range[0]
            rgb_max = color_range[1]
            seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max)) # 0 ~ 255
            seg_color = tuple(seg_color)
            while seg_color in self.used_seg_colors:
                seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max))
                seg_color = tuple(seg_color)
            seg_object = self._create_segmented_object(cop_obj_visible, seg_color)
            seg_object.set_parent(cop_obj)
            self.used_seg_colors.append(seg_color)
        else:
            seg_color = None
            seg_object = None
        if cn_tag == -1: 
            self.scene_furnitures.append((cop_obj, cop_obj_visible, seg_object, seg_color, obj_rot_axis))
        else:
            self.scene_connectors.append((cop_obj, cop_obj_visible, seg_object, seg_color, obj_rot_axis))

        self._randomize_obj_pose(cop_obj, obj_rot_axis, z_max) #TODO:

        self.pr.step()

    def _get_base_objects_path(self):
        #TODO:
        part_path = join(self.current_path, "part" + str(self.process_id)) 
        furnitures_path = join(part_path, "furniture_part")
        connectors_path = join(part_path, "connector_part")
        furniture_f_list = get_file_list(furnitures_path)
        connector_f_list = get_file_list(connectors_path)
        furniture_list = []
        connector_list = []
        furniture_names = []
        connector_names = []
        furniture_textures = []
        connector_textures = []
        for file_path in furniture_f_list:
            part_name, file_ext = os.path.splitext(file_path)
            if 'obj' in file_ext:
                furniture_names.append(part_name.replace(furnitures_path, ""))
                furniture_list.append(file_path)
            elif 'original' in part_name:
                furniture_textures.append(file_path)
            else:
                pass

        for file_path in connector_f_list:
            part_name, file_ext = os.path.splitext(file_path)
            if 'obj' in file_ext:
                connector_names.append(part_name.replace(connectors_path, ""))
                connector_list.append(file_path)
            elif 'original' in part_name:
                connector_textures.append(file_path)
            else:
                pass

        furniture_list.sort()
        furniture_textures.sort()
        connector_list.sort()
        connector_textures.sort()

        self.color_range_furn = 255 / len(furniture_list)
        self.color_range_conn = 255 / len(connector_list)
        self.class_colors = {}
        self.class_num = len(connector_list) + len(furniture_list) # objects to segmentation

        for class_id in range(self.class_num): # 5
            if class_id < len(furniture_list):
                rgb_min = [self.color_range_furn * class_id + 1, 0, 0]
                rgb_max = [self.color_range_furn * (class_id + 1), 0, 0]
                self.class_colors[class_id] = (rgb_min, rgb_max)
            else:
                rgb_min = [0, self.color_range_conn * (class_id - len(furniture_list)) + 1, 0]
                rgb_max = [0, self.color_range_conn * (class_id - len(furniture_list) + 1), 0]
                self.class_colors[class_id] = (rgb_min, rgb_max)

        furnitures = []
        for furniture_id, obj in enumerate(furniture_list):
            class_id = furniture_id
            furnitures.append((obj, class_id, furniture_textures[0]))
        # import connectors:
        connectors = []
        for connector_id, obj in enumerate(connector_list):
            class_id = connector_id + len(furniture_list)
            connectors.append((obj, class_id, connector_textures[connector_id]))

        print("[INFO] successfully read {} furnitures".format(len(furnitures)))
        print("[INFO] successfully read {} connectors".format(len(connectors)))
        print("[INFO] segmentation color setting is: \n")
        for i, fn in enumerate(furniture_names):
            print("\t Furniture: {} Class_ID: {}".format(fn, furnitures[i][1]))
        for i, fn in enumerate(connector_names):
            print("\t Connector: {} Class_ID: {}".format(fn, connectors[i][1]))
        for idx, class_range in self.class_colors.items():
            print("\t Class_ID: {} object has {} ~ {} color range".format(idx, class_range[0], class_range[1]))
        print("\n")

        return (furnitures, connectors)

    def _import_random_object(self, cn_tag=-1):
        if cn_tag == -1:
            print("[INFO] import furniture / ", datetime.datetime.now())
            obj_path, class_idx, tex_path = random.choice(self.base_path[0])
            scaling_factor = 1.0
        else:
            print("[INFO] import connector / ", datetime.datetime.now())
            obj_path, class_idx, tex_path = self.base_path[1][cn_tag]
            scaling_factor = 0.001
        self._set_texture_grad(tex_path)
        cop_obj_visible = Shape.import_shape(filename= obj_path,
                                             scaling_factor=scaling_factor,
                                            )
        try:
            cop_obj = cop_obj_visible.get_convex_decomposition()
        except:
            cop_obj = cop_obj_visible.copy()
        cop_obj_visible.set_parent(cop_obj)
        cop_obj_visible.set_renderable(True)
        cop_obj.set_parent(self.target_base)
        cop_obj.set_renderable(False)

        if 'short' in obj_path or 'long' in obj_path:
            tag = True
        else:
            tag = False
        z_max, obj_rot_axis = self._get_rotation_axis(cop_obj, tag)

        # select segmentation color for object and create segmentation texture
        if not class_idx == -1: # -1 => no segmentation
            color_range = self.class_colors[class_idx]
            # obj_idx = class_color[0]
            # if not class_idx == obj_idx:
            #     print("[ERROR] class index is not matching to class color")
            # color_range = class_color[1]
            rgb_low = color_range[0]
            rgb_max = color_range[1]
            seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max)) # 0 ~ 255
            seg_color = tuple(seg_color)
            while seg_color in self.used_seg_colors:
                seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max))
                seg_color = tuple(seg_color)
            seg_object = self._create_segmented_object(cop_obj_visible, seg_color)
            seg_object.set_parent(cop_obj)
            self.used_seg_colors.append(seg_color)
        else:
            seg_color = None
            seg_object = None
        if cn_tag == -1:
            self.scene_furnitures.append((cop_obj, cop_obj_visible, seg_object, seg_color, obj_rot_axis))
        else:
            self.scene_connectors.append((cop_obj, cop_obj_visible, seg_object, seg_color, obj_rot_axis))

        self._randomize_obj_pose(cop_obj, obj_rot_axis, z_max) #TODO:

        self.pr.step()


    def _initialize_robots(self):
        if self.env_tag == 0: # gist
            self.robot_num = 1
        else:
            self.robot_num = 2
        self.robot_color = (0, 0, 100)
        self.robots = []
        for i in range(self.robot_num):
            self.robots.append(MyPanda(i))
        for robot in self.robots:
            robot_visible_parts = robot.get_visible_objects()
            for visible_part in robot_visible_parts:
                seg_part = self._create_segmented_object(visible_part, self.robot_color)
                seg_part.set_parent(visible_part)
                robot.add_seg_parts(seg_part)

    #endregion

    #region setting object pose    
    def _get_rotation_axis(self, shape, tag=False):
        bbox = shape.get_bounding_box()
        xyz = [bbox[1], bbox[3], bbox[5]]
        # find align info
        if tag:
            val, idx = max((val, idx) for (idx, val) in enumerate(xyz))
            idx += np.random.randint(1, 3)
            idx = idx % 3
            val = xyz[idx]
        else:
            val, idx = min((val, idx) for (idx, val) in enumerate(xyz))

        shape.set_orientation([0, 0, 0])
        self.pr.step()
        if idx == 0: # x is min
            shape.rotate([0, np.pi / 2, 0])
        elif idx == 1: # y is min
            shape.rotate([np.pi / 2, 0, 0])
        else: # z is min
            pass
        self.pr.step()
        rot_axis = idx
        
        return val, rot_axis
    
    def _collision_check2collision_wall(self, obj):
        collision_state = False
        for wall in self.invisible_walls:
            if obj.check_collision(wall):
                collision_state = True
                return collision_state
        
        return collision_state

    def _randomize_obj_pose(self, obj, rot_axis, z_max):#TODO: set position in workspace
        """randomize object pose within box area
        Arguments:
            obj {[type]} -- [description]
            rot_axis {} -- [rotation axis for object]
        """
        # print("[INFO] set object property /", datetime.datetime.now())
        obj.set_collidable(True)
        obj.set_dynamic(False)
        obj.set_respondable(False)
        # print("[INFO] set object position /", datetime.datetime.now())
        random_position = list(np.random.uniform(self.workspace_range[0], self.workspace_range[1]))
        random_position += [z_max - 0.3]
        obj.set_position(random_position, relative_to=self.workspace)
        # print("[INFO] set object orientation /", datetime.datetime.now())
        random_rotation = [0, 0, 0]
        random_value = np.random.uniform(-np.pi, np.pi, 1)
        random_rotation[rot_axis] = random_value
        random_rotation[rot_axis-1] = np.pi if np.random.rand() > 0.5 else 0
        obj.rotate(random_rotation)
        self.pr.step()

        collision_state = True
        count = 0
        while collision_state:
            if not self._collision_check2collision_wall(obj):
                collision_state = False
            if not z_max == 0.1: # for furniture
                if len(self.scene_furnitures) > 1:
                    for obj_info in self.scene_furnitures:
                        if obj == obj_info[0]:
                            continue
                        else:
                            if obj.check_collision(obj_info[0]):
                                collision_state = True
                                break
                            else:
                                pass
            if not collision_state:
                break
            count += 1
            if count > 5:
                z_max = 0.1
            # print("[INFO] set object position /", datetime.datetime.now())
            random_position = list(np.random.uniform(self.workspace_range[0], self.workspace_range[1]))
            random_position += [z_max-0.3]
            obj.set_position(random_position, relative_to=self.workspace)
            # print("[INFO] set object orientation /", datetime.datetime.now())
            random_rotation = [0, 0, 0]
            random_value = np.random.uniform(-np.pi, np.pi, 1)
            random_rotation[rot_axis] = random_value
            random_rotation[rot_axis - 1] = np.pi if np.random.rand() > 0.5 else 0
            obj.rotate(random_rotation)
            self.pr.step()
        
        obj.set_dynamic(True)
        obj.set_respondable(True)
        self.pr.step() 

    def _set_scene_obj_dynamics(self, dynamic):
        for obj_info in self.scene_furnitures:
            obj_info[0].set_dynamic(dynamic)
        for obj_info in self.scene_connectors:
            obj_info[0].set_dynamic(dynamic)
        
        self.pr.step()

    #endregion

    #region segmentation setting
    def _create_segmented_object(self, obj_visible, seg_color):
        seg_object = obj_visible.copy()
        seg_object.set_color([1, 1, 1])
        try:
            seg_object.remove_texture()
        except:
            pass
        seg_texture = self._create_segmentation_textures(seg_color)
        seg_object.set_texture(texture=seg_texture,
                               decal_mode=True,
                               mapping_mode=TextureMappingMode(3),
                               uv_scaling=self._texture_uv,
                               repeat_along_u=True,
                               repeat_along_v=True,
                               )
        seg_object.set_renderable(False)
        self.pr.step()

        return seg_object

    def _create_segmentation_textures(self, color):
        """get list of segmentation colors for objects in scene and create each color texture
        and each texture get from the shape(plane)
        Arguments:
            colors {tuple(3)[0-255]} -- [description]
        Returns:
            texture {texture} -- [texture of created shape(not work when shape removed)]
        """
    
        texture_path = get_segmentation_image(color, self.process_id)
        shape, texture = self.pr.create_texture(filename=texture_path,
                                          repeat_along_u=True, 
                                          repeat_along_v=True)
        
        self.texture_shapes.append(shape)
        
        return texture

    def _set_segmentation_textures(self):
        """set each objects texture to segmentation color
        should be proceed after create segmentation textures
        """
        for obj_info in self.scene_furnitures:
            obj_info[1].set_texture(texture=obj_info[3],
                                    decal_mode=True,
                                    mapping_mode=TextureMappingMode(3),
                                    uv_scaling=self._texture_uv,
                                    repeat_along_u=True,
                                    repeat_along_v=True,
                                    )
        for obj_info in self.scene_connectors:
            obj_info[1].set_texture(texture=obj_info[3],
                                    decal_mode=True,
                                    mapping_mode=TextureMappingMode(3),
                                    uv_scaling=self._texture_uv,
                                    repeat_along_u=True,
                                    repeat_along_v=True,
                                    )  
        self.pr.step()

    def _set_segmented_objects_render(self, render):
        for obj_info in self.scene_furnitures:
            obj_info[2].set_renderable(render)
        for obj_info in self.scene_connectors:
            obj_info[2].set_renderable(render)
        
        for robot in self.robots:
            robot.set_segmentation_objects_render(render)

        self.seg_table.set_renderable(render)

        self.pr.step()

    #endregion
    
    #region texture and color randomize

    def _get_random_color(self, refer=None):
        """[get random color]
        
        Keyword Arguments:
            refer {[list(3)(0-1)]} -- [reference color] (default: {None})
        """
        if not type(refer) == list: # no reference color
            return list(np.random.rand(3))
        else:
            reference_color = np.array(refer)
            color_min = np.clip(reference_color - self._color_variance, 0,1) 
            color_max = np.clip(reference_color + self._color_variance, 0,1)
            random_color = list(np.random.uniform(color_min, color_max))
            return random_color

    def _get_gray_texture_files(self):
        """get all gray texture file list
        
        Returns:
            [type] -- [description]
        """
        texture_path = self.current_path + "/gray_textures/"
        texture_list = get_file_list(texture_path)

        return texture_list
 
    def _get_hard_texture_files(self):
        """get all hard texture file list
        
        Returns:
            [type] -- [description]
        """
        texture_path = self.current_path + "/hard_textures/"
        # texture_path = "/IITP/seung/AugmentedAutoencoder/VOCdevkit/VOC2012/JPEGImages"
        texture_list = get_file_list(texture_path)

        return texture_list
    
    def _get_random_gray_texture(self):
        """get random gray texture from texture list
        
        Returns:
            texture file path
        """
        ind = np.random.randint(len(self.gray_textures))
        texture = self.gray_textures[ind]
        
        return texture
    
    def _get_random_hard_texture(self):
        """get random gray texture from texture list
        
        Returns:
            texture file path
        """
        ind = np.random.randint(len(self.hard_textures))
        texture = self.hard_textures[ind]
        
        return texture

    def _set_texture_grad(self, texture_path, refer=None):
        first_color = self._get_random_color(refer=refer)
        if refer==None:
            refer = list(1 - np.array(first_color)) # complement color
        second_color = self._get_random_color(refer=refer)
        set_texture_grad(texture_path, first_color, second_color)
    
    def _get_grad_randomized_texture(self, refer=None):
        """create randomize texture in scene
        1. get random gray texture from texture file list
        2. get two random color(can be refer)
        3. create gradation texture image
        4. create texture shape(in the scene) 
        
        Keyword Arguments:
            refer {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        texture_path = self._get_random_gray_texture()
        first_color = self._get_random_color(refer=refer)
        if refer==None:
            refer = list(1 - np.array(first_color)) # complement color
        second_color = self._get_random_color(refer=refer)
        rand_texture = create_randomize_image(texture_path,
                                              first_color,
                                              second_color,
                                              self.process_id)        
        while not type(rand_texture) == str:
            print("[ERROR] {} texture occur error!".format(texture_path))
            texture_path = self._get_random_gray_texture()
            first_color = self._get_random_color(refer=refer)
            second_color = self._get_random_color(refer=refer)
            rand_texture = create_randomize_image(texture_path,
                                                first_color,
                                                second_color,
                                                self.process_id)        
                
        shape, texture = self.pr.create_texture(filename=rand_texture,
                                                repeat_along_u=True, 
                                                repeat_along_v=True)
        self.texture_shapes.append(shape)
        # self.pr.step()

        return texture
    
    def _get_hard_randomized_texture(self):
        texture_path = self._get_random_hard_texture()
        shape, texture = self.pr.create_texture(filename=texture_path,
                                                repeat_along_u=True, 
                                                repeat_along_v=True)
        self.texture_shapes.append(shape)
        self.pr.step()

        return texture
    


    def _randomize_object_texture(self, obj_visible):
        texture = self._get_grad_randomized_texture()
        mapping_index = np.random.randint(4)
        texture_uv = list(np.random.rand(2))
        obj_visible.set_texture(texture=texture,
                                mapping_mode=TextureMappingMode(mapping_index),
                                uv_scaling=texture_uv,
                                repeat_along_u=True,
                                repeat_along_v=True,
                                )

    def _get_obj_textures(self, obj_list):
        """get current textures from scene objects
        each textures saved and used after segmentation
        Returns:
            textures [texture] -- [current texture of scene object]
        """
        textures = []
        for obj_info in obj_list:
            try:
                texture = obj_info[1].get_texture()
                textures.append(texture)
            except: # object has no texture
                textures.append(obj_info[1].get_color())
    
        return textures

    def _set_obj_textures(self, obj_list, textures):
        """get saved textures and set each objects' texture
        
        Arguments:
            textures {[type]} -- [description]
        """
        for texture, obj_info in zip(textures, obj_list):
            rand_ori = list(np.random.uniform(0, 1, 3))
            uv_scaling = list(np.random.rand(2))
            mapping_mode = TextureMappingMode(np.random.randint(4))
            obj_info[1].set_texture(texture=texture,
                                    mapping_mode=mapping_mode,
                                    uv_scaling=uv_scaling,
                                    orientation=rand_ori,
                                    repeat_along_u=True,
                                    repeat_along_v=True,
                                    )
        self.pr.step()
    
    def _set_visible_object_render(self, render):
        for obj_info in self.scene_furnitures:
            obj_info[1].set_renderable(render)
        for obj_info in self.scene_connectors:
            obj_info[1].set_renderable(render)
        
        for robot in self.robots:
            robot.set_visible_object_render(render)
        self.table.set_renderable(render)

        self.pr.step()

    #endregion

    #region randomize scene(except target object)
    def _randomize_floor(self):
        bottom = self.floors[0]
        walls = self.floors[1:]
        bottom_texture = self._get_hard_randomized_texture()
        wall_texture = self._get_hard_randomized_texture()
        bottom.set_texture(texture=bottom_texture,
                           mapping_mode=TextureMappingMode(0),
                           uv_scaling=[3, 3],
                           repeat_along_u=True,
                           repeat_along_v=True)
        for wall in walls:
            wall.set_texture(texture=wall_texture,
                             mapping_mode=TextureMappingMode(0),
                             uv_scaling=[3, 3],
                             repeat_along_u=True,
                             repeat_along_v=True)

    def _randomize_table(self):
        table_texture = self._get_grad_randomized_texture()
        self.table.set_texture(texture=table_texture,
                               mapping_mode=TextureMappingMode(3),
                               uv_scaling=[3, 3],
                               repeat_along_u=True,
                               repeat_along_v=True)       

    def _randomize_camera(self): #TODO:
        """randomize camera start position, angle
        set random camera distance and elevation
        set random perspective angle
        """
        # reset camera pose
        

        # randomize z distance
        rand_z = np.random.uniform(self._min_z, self._max_z)
        self.camera_control.set_position([0, 0, rand_z], relative_to=self.camera_rot_base)
        self.pr.step()

        # randomize elevation and z rotation
        
        # randomize azimuth
        z_rot = np.random.uniform(self._azimuth_range[0], self._azimuth_range[1])
        

        # randomize perspective angle
        rand_pangle = np.random.uniform(self._min_pangle, self._max_pangle)
        self.main_camera.set_perspective_angle(rand_pangle)
        self.target_camera.set_perspective_angle(rand_pangle)
        
        self.pr.step()

    def _randomize_light(self):
        for light in self.lights:
            light.light_off()
        self.n_light = np.random.randint(1, self._max_n_light + 1)
        index_light = np.random.randint(self._max_n_light, size=self.n_light)
        for index in index_light:
            random_diffuse = list(np.random.rand(3))
            random_specular = list(np.random.rand(3))
            self.lights[index].light_on(random_diffuse,random_specular)    

    def _randomize_robot(self):
        for robot in self.robots:
            robot.set_random_pose()
            for part in robot.visible_parts:
                robot_texture = self._get_grad_randomized_texture()
                part.set_texture(texture=robot_texture,
                                 mapping_mode=TextureMappingMode(3),
                                 uv_scaling=[3, 3],
                                 repeat_along_u=True,
                                 repeat_along_v=True)


            #endregion

    def randomize(self):
        print("[INFO] randomize enviroment / ", datetime.datetime.now())

        self._randomize_light()
        self._randomize_floor()
        self._randomize_table()
        self._randomize_robot()
        self.pr.step()

    def reset(self, furniture_num, connector_min, connector_max):
        print("[RESET] reset environment / ", datetime.datetime.now())
        self.randomize()
        

        for obj_info in self.scene_furnitures:
            obj = obj_info[0]
            obj_vis = obj_info[1]
            seg_obj = obj_info[2]
            obj.remove()
            obj_vis.remove()
            seg_obj.remove()
        for obj_info in self.scene_connectors:
            obj = obj_info[0]
            obj_vis = obj_info[1]
            seg_obj = obj_info[2]
            obj.remove()
            obj_vis.remove()
            seg_obj.remove()
            
        self.scene_furnitures = []
        self.scene_connectors = []
        
        for shape in self.texture_shapes:
            shape.remove()
        self.texture_shapes = []
        self.used_seg_colors = []
        self.pr.step()

        for i in range(furniture_num):
            self._import_random_object()
        for i in range(len(self.base_path[1])):
            connector_num = np.random.randint(connector_min, connector_max+1)
            for j in range(connector_num):
                self._import_random_object(cn_tag=i)

        self.pr.step()

    def step(self, add_object=False):
        self._randomize_camera()

        self.pr.step()
    
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_images(self):
        """[get images from scene]
        main_camera: view all renderable objects in scene
        target_camera: view renderable objects in taget(collection)
        1. rgb, depth image of scene
        2. segmented image, mask image of target objects
        """
        # fix objects pose
        self._set_scene_obj_dynamics(False)
        # check object in workspace 
        for obj_info in self.scene_furnitures:
            obj = obj_info[0]
            cur_z = obj.get_position()[2]
            if cur_z < 1:
                obj.set_parent(None)
        for obj_info in self.scene_connectors:
            obj = obj_info[0]
            cur_z = obj.get_position()[2]
            if cur_z < 1:
                obj.set_parent(None)

        # 1. rgb, depth image        
        main_rgb, main_depth = self.main_camera.get_image()
        target_rgb, _ = self.target_camera.get_image()
        # 2. segmentation image and mask image
        # 2.1 set render setting
        self._set_visible_object_render(False)
        self._set_segmented_objects_render(True)
        
        # 2.2 get segmentation image and mask
        objects_seg, objects_depth = self.target_camera.get_image()
        objects_mask = objects_depth < 0.99
        #region occulusion        
        """
        print("[INFO] remove occulusion objects / ", datetime.datetime.now())
        for obj, seg_color in zip(self.scene_objects, self.seg_colors):
            obj.set_parent(self.target_base2)
            self.pr.step()
            _, obj_depth = self.target2_camera.get_image()
            obj_mask = np.where(obj_depth < 0.99, 1, 0)
            obj_pixel_num = np.sum(obj_mask) 
            
            seg_color = np.array(seg_color) / 255
            not_occ_tag = np.where(np.abs(objects_seg - seg_color) < 1/255 , True, False)
            # not_occ_tag = np.where(np.abs(np.uint8(objects_seg * 255) - seg_color) <= 1, True, False)
            not_occ_tag = np.all(not_occ_tag, axis=2)
            obj_mask[not_occ_tag] = 0
            occ_num = np.sum(obj_mask) 
            
            if occ_num > (obj_pixel_num * 4 / 5):
                objects_mask[not_occ_tag] = 0
                objects_seg[not_occ_tag] = (0, 0, 0)
            obj.set_parent(self.target_base)
            self.pr.step()
        """
        #endregion
        # 3. set render setting
        self._set_visible_object_render(True)
        self._set_segmented_objects_render(False)
        
        # self._set_scene_obj_dynamics(True)

        objects_pose = {}
        for obj_info in (self.scene_connectors + self.scene_furnitures):
            objects_pose[obj_info[3]] = obj_info[0].get_pose(relative_to=self.camera_control)
        objects_pose['camera_position'] = self.camera_control.get_position(relative_to=self.camera_rot_base)

        return main_rgb, main_depth, target_rgb, objects_seg, objects_mask, objects_pose #TODO:(jsjs)
