from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim
import numpy as np
from scipy.spatial.transform import Rotation as R

def degree2radian(degree):
    return (degree/180)*np.pi

class MyCamera(object):
    """Azure Kinect spec
    depth is 1:1 resolution
    rgb is 4:3 resolution

    """
    def __init__(self, name, resolution, perspective_angle=90):
        self.base = Dummy(name)
        self.base_name = self.base.get_name().replace("_camera","")
        self.rgb = VisionSensor(self.base_name + "_rgb") 
        self.depth = VisionSensor(self.base_name + "_depth")
        self._initial_pose = self.base.get_pose()
        # camera intrinsic
        self._set_resolution(resolution)
        self._perspective_angle = perspective_angle
    
    def _set_resolution(self, resolution):
        self.rgb.set_resolution(resolution)
        self.depth.set_resolution(resolution)
        self._resolution = resolution

    def get_resolution(self):
        return self._resolution

    def get_pose(self, relative_to=None):
        """
        return pose[x, y, z, qx, qy, qz, w]
        """
        try:
            if relative_to == None:
                return self.base.get_pose()
            else:
                return self.base.get_pose(relative_to=relative_to)

        except :
            return self.base.get_pose(relative_to=relative_to)

    def get_position(self, relative_to=None):
        """
        return position[x, y, z]
        """
        try:
            if relative_to == None:
                return self.base.get_position()
            else:
                return self.base.get_position(relative_to=relative_to)

        except :
            return self.base.get_position(relative_to=relative_to)

    def set_position(self, position, relative_to=None):
        try:
            if relative_to == None:
                self.base.set_position(position)
            else:
                self.base.set_position(position, relative_to=relative_to)
        except:
            self.base.set_position(position, relative_to=relative_to)

    def rotate_rel(self, relto, rotation):
        """[rotate camera]
        unlike set_position or set_orientation 
        camera pose should be reset to base after rotate base
        Arguments:
            relto {[Object]} -- [rotation center]
            rotation {[list]} -- [rotation degree]
        """
        M = self.base.get_matrix()
        m = relto.get_matrix()
        x_axis = [m[0], m[4], m[8]]
        y_axis = [m[1], m[5], m[9]]
        z_axis = [m[2], m[6], m[10]]
        pos = [m[3], m[7], m[11]]
        M = sim.simRotateAroundAxis(M, z_axis, pos, rotation[2])
        M = sim.simRotateAroundAxis(M, y_axis, pos, rotation[1])
        M = sim.simRotateAroundAxis(M, x_axis, pos, rotation[0])
        self.base.set_matrix(M)

    def set_perspective_angle(self, angle):
        self.rgb.set_perspective_angle(angle)
        self.depth.set_perspective_angle(angle)
        self._perspective_angle = angle

    def get_perspective_angle(self):
        return self._perspective_angle

    def get_image(self):
        """get images from vision sensor in scene
        depth image is 0 ~ 1 and 0.998... is not detected value

        Returns:
        """
        rgb_image = self.rgb.capture_rgb()
        depth_image = self.depth.capture_depth()
        
        return rgb_image, depth_image

    def reset(self):
        self.base.set_pose(self._initial_pose)
