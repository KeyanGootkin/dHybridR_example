from pysim.utils import yesno
from pysim.parsing import Folder
from pysim.environment import simulationDir


class GenericSimulation:
    def __init__(
            self, 
            path:str, 
            template:str|Folder=None,
            caching:bool=False,
            verbose:bool=True,
        ) -> None:
        self.template = template
        self.verbose = verbose
        #setup cache
        if self.verbose: print("caching is ON..." if caching else "caching is OFF...")
        self.caching = caching 
        self.cache: dict = {}
        #make sure the simulation exists
        if verbose: print(f"Finding path: {path}")
        self.path: str = path
        self.dir = Folder(path)
        self.name = self.dir.name
        #if the given path doesn't exist, check the default simulation directory 
        if not self.dir.exists(): 
            if self.verbose: print(f"No simulation found in {path}, checking default simulation directory: {simulationDir.path}...")
            default_path: str = simulationDir.path+self.name
            self.path = default_path
            self.dir = Folder(self.path)
            #if simulation isn't in default simulation directory either copy a template to that location or raise an error
            if not self.dir.exists():
                if yesno(f"No such simulation exists, would you like to copy \ntemplate: {template.name}, \nto location: {self.path}?\n"):
                    self.create()
                else: raise FileNotFoundError("Please create simulation and try again")
    
    def create(self):
        self.template.copy(self.path)
