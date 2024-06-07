from pysim.parsing import Folder
from pysim.utils import progress_bar
from glob import glob

master = Folder("/home/x-kgootkin/pysim")

scratch = "/anvil/scratch/x-kgootkin"
home = "/home/x-kgootkin"

locations = [Folder(home+"/turbulence/pysim", master=master), Folder(home+"/flarewave/pysim", master=master)] +\
            [Folder(s+"/pysim", master=master) for s in glob(scratch+"/sims/*")]

def update():
    for l in progress_bar(locations, total=len(locations)): 
        print(f"updating {l.path}...")
        l.update()


if __name__=="__main__": 
    update()