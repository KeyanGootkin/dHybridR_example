import pysim as sim
from glob import glob
import os
import shutil
import numpy as np

amp = 1.,
k = 1., np.pi

if __name__ == '__main__':
    # clean files
    for file in glob('input/*.unf'):
        os.remove(file)
    if os.path.exists("Output"): shutil.rmtree("Output")
    if os.path.exists("Restart"): shutil.rmtree("Restart")
    # make init files
    s = sim.dHybridR("test")
    Init = sim.FlareWaveInit(s)
    Init.build_B_field()
    Init.save_init_field(Init.B, path='input/Bfld.unf')
    os.system("sh submit_anvil.sh")