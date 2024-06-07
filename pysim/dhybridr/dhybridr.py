#pysim imports
from pysim.utils import yesno 
from pysim.parsing import Folder
from pysim.environment import dHybridRtemplate
from pysim.fields import ScalarField, VectorField
from pysim.simulation import GenericSimulation
from pysim.dhybridr.input import dHybridRinput
from pysim.dhybridr.initializer import dHybridRinitializer
from pysim.dhybridr.anvil_submit import AnvilSubmitScript
#nonpysim imports
import numpy as np 
from h5py import File as h5File
from os import system

# simulation parsing
def extract_energy(file_name: str) -> tuple:
    with h5File(file_name, 'r') as file:
        fE = np.mean(file["DATA"], axis=1)
        [low, high] = file["AXIS"]["X2 AXIS"][:]
        lne = np.linspace(low, high, fE.shape[0])
        dlne = np.diff(lne)[0]
        E = np.exp(lne)
        return E, fE, dlne


class dHybridR(GenericSimulation):
    """
    A simulation class to interact with dHybridR simulations in python
    """
    def __init__(
            self, 
            path: str,
            caching: bool = False,
            verbose: bool = False,
            template: Folder = dHybridRtemplate
        ) -> None:
        #setup simulation
        GenericSimulation.__init__(self, path, caching=caching, verbose=verbose, template=template)
        #setup input, output, and restart folders
        self.parse_input()
        self.outputDir = Folder(self.path+"/Output")
        if not self.outputDir.exists():
            if yesno("There is no output, would you like to run this simulation?\n"): 
                self.run()
        else: self.parse_output()
        self.restartDir = Folder(self.path+"/Restart")
    def __repr__(self) -> str: return self.name
    def create(self) -> None:
        self.template.copy(self.path)
        system(f"chmod 755 {self.path}/dHybridR")
    def parse_input(self) -> None:
        self.input = dHybridRinput(self.path+"/input/input")
        self.dt = self.input.dt
        self.niter = self.input.niter
        self.dx: float = self.input.boxsize[0]/self.input.ncells[0]
        self.dy: float = self.input.boxsize[1]/self.input.ncells[1]
    def run(self, initializer: dHybridRinitializer, submit_script: AnvilSubmitScript) -> None:
        initializer.prepare_simulation()
        submit_script.write()
        system(f"sh {submit_script.path}")
    def parse_output(self) -> None:
        kwargs = {'caching':self.caching, 'verbose':self.verbose, 'parent':self}
        self.B       = VectorField(self.path + "/Output/Fields/Magnetic/Total/", name="magnetic", latex="B", **kwargs)
        self.E       = VectorField(self.path + "/Output/Fields/Electric/Total/", name="electric", latex="E", **kwargs)
        self.etx1    = ScalarField(self.path + "/Output/Phase/etx1/Sp01/", **kwargs)
        self.pxx1    = ScalarField(self.path + "/Output/Phase/p1x1/Sp01/", **kwargs)
        self.pyx1    = ScalarField(self.path + "/Output/Phase/p2x1/Sp01/", **kwargs)
        self.pzx1    = ScalarField(self.path + "/Output/Phase/p3x1/Sp01/", **kwargs)
        self.density = ScalarField(self.path + "/Output/Phase/x3x2x1/Sp01/", name="density", latex=r"$\rho$", **kwargs)
        self.Pxx     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/xx/", **kwargs)
        self.Pyy     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/yy/", **kwargs)
        self.Pzz     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/zz/", **kwargs)
        self.u       = VectorField(self.path + "/Output/Phase/FluidVel/Sp01/", name="bulkflow", latex="u", **kwargs)
        [
            self.energy_grid,
            self.energy_pdf,
            self.dlne
        ] = np.array([[*extract_energy(f)] for f in self.etx1.file_names], dtype=object).T