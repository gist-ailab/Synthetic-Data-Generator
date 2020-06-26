from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape

import numpy as np

class MyPanda(object):
    def __init__(self, instance=0):
        self.arm = Panda(instance)
        self.gripper = PandaGripper(instance)
        self.initial_position = self.arm.get_position()
        self.initial_joint_positions = self.arm.get_joint_positions()
        self.visible_parts = self._extract_visible_parts()
        
    def _extract_visible_parts(self):
        visible_parts = []
        visible_arm = self.arm.get_visuals()
        visible_grip = self.gripper.get_visuals()
        for v_a in visible_arm:
            visible_parts += v_a.ungroup()
        for v_g in visible_grip:
            visible_parts += v_g.ungroup()

        return visible_parts
