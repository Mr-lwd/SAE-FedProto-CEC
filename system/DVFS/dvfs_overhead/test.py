import ctypes 
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'change_config_noprint.so')
        
cLib = ctypes.CDLL(lib_path)

start=time.perf_counter()
cLib.changeCpuFreq(345600)
overhead=time.perf_counter() - start

print("overhead", overhead)
time.sleep(2)

cLib.changeCpuFreq(2035200)