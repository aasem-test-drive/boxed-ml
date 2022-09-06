import platform
import os
import json

class utilities:
    def __init__(self):
        self.supported_system=['Linux','Darwin','Windows']
        self.os_system=platform.system()
        #self.os_release=platform.release()

    def os_compatible(self):
        if (self.os_system in self.supported_system):
            return True
        else:
            return False
        
    def os_clearscreen(self):
        if self.os_system=='Windows':
            os.system('cls')
        else:
            os.system('clear')

    def gpu_check():
        #todo: check GPU
        print("PhysicalDevice=tf.config.list_physical_devices('GPU')")
        
    def dump_env(env='conda'):
        if env=='conda':
            #todo: dump conda env to file
            print ('$ !conda list --explicit > $prefix/conda_env.txt')

    def load_JSON_file(self, path_to_file):
        try:
            obj = open(path_to_file)
            data = json.load(obj)        
            obj.close()
            retObj=json.loads(json.dumps(data))
        except Exception as e:
            retObj=e
        return retObj
