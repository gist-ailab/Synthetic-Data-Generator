from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.backend import sim
from pyrep.backend._sim_cffi import ffi, lib

class Light(Object):

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.LIGHT
    
    def light_on(self, diff, spec):
        ret = lib.simSetLightParameters(self._handle, 1, ffi.NULL, diff, spec)
        sim._check_return(ret)
    
    def light_off(self):
        ret = lib.simSetLightParameters(self._handle, 0, ffi.NULL, [0.5,0.5,0.5],[0.5,0.5,0.5])
        sim._check_return(ret)
    
    def simGetLightParameters(self):
        pass