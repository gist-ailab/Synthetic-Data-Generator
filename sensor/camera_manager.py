from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
from sensor.mycamera import MyCamera


class CameraManager(object):
    """
    move cameras
    get images
    manage targets
    """
    def __init__(self, resolution):
        self.controller = Dummy("camera_control")
        self.rot_base = Dummy("camera_rot_base")
        self.target_base = Dummy('target')        
        
        self._resolution = resolution
        self.main_camera = MyCamera("main_camera", resolution)
        self.target_camera = MyCamera("target_camera", resolution)

        self._initial_pose = self.controller.get_pose(relative_to=self.rot_base)

    def reset(self):
        #TODO: randomize rotation base
        self.controller.set_pose(self._initial_pose, relative_to=self.rot_base)
        
    def set_perspective_angle(self, angle):
        self.main_camera.set_perspective_angle(angle)
        self.target_camera.set_perspective_angle(angle)
        self._perspective_angle = angle

    def get_perspective_angle(self):
        
        return self._perspective_angle

    def get_resolution(self):
        
        return self.main_camera.get_resolution()

    def set_distance(self, distance):
        position = [0, 0, distance]
        self.controller.set_position(position, relative_to=self.rot_base)

    def set_rotation(self, elevation, azimuth, z_rotation):
        rotation = [0, elevation, z_rotation]
        self.rotate_rel(self.controller, rotation, self.rot_base)
        rotation = [0, 0, azimuth]
        self.rotate_rel(self.camera_control, rotation, self.rot_base)

    def add_target(self, target):
        target.set_parent(self.target_base)

    def capture(self):
        self.main_rgb, self.main_depth = self.main_camera.get_image()
        self.target_rgb, self.target_depth = self.target_camera.get_image()

    def get_main_rgb(self):
        return self.main_rgb
    def get_main_depth(self):
        return self.main_depth
    def get_target_rgb(self):
        return self.target_rgb
    def get_target_depth(self):
        return self.target_depth
    def get_target_mask(self):
        return self.target_depth > 0.99

    @staticmethod
    def rotate_rel(scene_object, rotation, relative_to):
        relto = relative_to
        M = scene_object.get_matrix()
        m = relto.get_matrix()
        x_axis = [m[0], m[4], m[8]]
        y_axis = [m[1], m[5], m[9]]
        z_axis = [m[2], m[6], m[10]]
        pos = [m[3], m[7], m[11]]
        M = sim.simRotateAroundAxis(M, z_axis, pos, rotation[2])
        M = sim.simRotateAroundAxis(M, y_axis, pos, rotation[1])
        M = sim.simRotateAroundAxis(M, x_axis, pos, rotation[0])
        scene_object.set_matrix(M)
