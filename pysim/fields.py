#pysim imports
import pysim.parsing as parsing
from pysim.parsing import Folder, File
from pysim.utils import verbose_bar
from pysim.plotting import show, show_video
#nonpysim imports
from glob import glob
import numpy as np
from h5py import File as h5File
from functools import cached_property
from os.path import isdir, isfile
import builtins
from matplotlib.cm import plasma as default_cmap

def curlz(X, Y, order: int = 2) -> np.ndarray:
    """
    take the z-component of the curl at each point in a field
    :param field: np.ndarray: [Fx, Fy] the x and y components of the vector field
    :param order: int: derivative order
    :return: c: np.ndarray: the curl of the field at each grid point
    """
    c = (
        (np.roll(Y, order, axis=1) - np.roll(Y, -order, axis=1)) -
        (np.roll(X, order, axis=0) - np.roll(X, -order, axis=0))
    ) / (2 * order)
    return c

def calc_psi(Bx, By, dx, dy):
    psi = np.zeros(Bx.shape)
    psi[1:,0] = np.cumsum(Bx[1:,0])*dy
    psi[:,1:] = (psi[:,0] - np.cumsum(By[:,1:], axis=1).T*dx).T
    return psi

class ScalarField:
    """
    a special class of arrays used to efficiently interact with the fields output by simulations
    ________
    ~Inputs~
    * source - str | array-like
        the source of the scalar field. Can be a file, a folder full of files, or an array like object.
    ___________
    ~Atributes~
    * single - bool
        whether or this is a single field as opposed to a collection of fields
    * shape - tuple[int]
        the shape of output arrays a la numpy arrays
    * ndims - int 
        the number of dimensions in the output arrays

    ===FILE MODE===
    * path - str
        path containing field files 
    * file_names - list[str]
        the files containing fields
    * caching - bool
        whether or not to store file outputs for later use, more memory intensive but fewer file accesses
    * cache - dict
        a dictionary to store file outputs for later use
    * reader - function
        the function used to read files

    ===ARRAY MODE===
    * array - numpy.ndarray
        the array representing the field
    """
    def __init__(
        self, 
        source: str|Folder|File|list|np.ndarray, 
        parent = None,
        caching: bool = False,
        verbose: bool = False,
        name: str = None, 
        latex: str = None
    ) -> None:
        self.name = name 
        self.latex = latex
        self.verbose = verbose
        self.parent = parent
        #setup cache
        self.caching = caching
        self.cache: dict = {}
        #find the correct constructor
        match type(source):
            #if its a folder
            case parsing.Folder:
                self.single = False
                example_file = File(source.children[0])
                if example_file.extension=="h5": self._from_folder_of_h5(source.path)
            case builtins.str if isdir(source): 
                self.single = False
                files = glob(source+"/*")
                extension = files[0].split(".")[-1]
                if extension=="h5": self._from_folder_of_h5(source)
            #otherwise its a file
            case parsing.File:
                self.single = True
                if source.extension=="h5": self._from_h5(source.path)
            case builtins.str if isfile(source): 
                self.single = True
                extension = source.split(".")[-1] 
                if extension=="h5": self._from_h5(source)
                else: self._from_csv(source)
            #or if its already been read
            case np.ndarray: 
                self.single = True
                self._from_numpy(source)
            case builtins.list: 
                self.single = True
                self._from_numpy(np.array(source))
    def __len__(self) -> int: return 1 if self.single else len(self.file_names)
    def __iter__(self):
        assert not self.single, "Cannot iterate through single scalar field"
        self.index = 0
        return self
    def __next__(self):
        if self.index < len(self):
            i = self.index
            self.index += 1
            return self[i]
        else: raise StopIteration
    def __getitem__(self, item: int|slice|tuple|list) -> np.ndarray:
        if self.single: return self.array[item]
        match type(item):
            case builtins.int: 
                return self.cache[item] if self.caching and item in self.cache.keys() else self.reader(self.file_names[item], item)
            case builtins.slice:
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([
                    self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item_iters
                ])
            case builtins.tuple|builtins.list: return np.array([
                self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item
            ])

    def _from_folder_of_h5(self, path:str) -> None: 
        self.path = path.path if isinstance(path, Folder) else path
        self.file_names: list = sorted(glob(path + "/*.h5"))
        self.reader: function = self._read_h5_file
        self.shape = self[0].shape
        self.ndims = len(self.shape)
    def _from_h5(self, file:str) -> None:
        self.file = file.path if isinstance(file,File) else file
        self.array = self._read_h5_file(file, 0)
        self.shape = self.array.shape 
        self.ndims = len(self.shape)
    def _read_h5_file(self, file:str, item) -> np.ndarray:
        with h5File(file, 'r') as f:
            output = np.array(f["DATA"][:])
            #GODDMANIT I HATE THAT IT DOES Y,X and not X,Y
            output = output.transpose((1,0))
            if self.caching: self.cache[item] = output
            return output
    
    def _from_csv(self) -> None:
        self.single = True
    def _read_csv_file(self) -> None: pass
    
    def _from_numpy(self, array:np.ndarray) -> None:
        self.single = True
        self.array = array
        self.shape = array.shape
        self.ndims = len(self.shape)

    def show(self, item:int, **kwargs) -> None: show(self[item],**kwargs)
    
    def movie(self, norm='none', cmap=default_cmap, alter_func=None,**kwrg) -> None:
        @show_video(name=self.name, latex=self.latex, norm=norm, cmap=cmap)
        def reveal_thyself(s,alter_func=alter_func, **kwargs): 
            return np.array([self[i] for i in range(len(self))]) if alter_func is None else np.array([alter_func(self[i]) for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, alter_func=alter_func,**kwrg)

class VectorField:
    def __init__(
            self, 
            *components, 
            caching: bool =False, 
            verbose: bool =False,
            name: str = None, 
            latex: str = None,
            parent = None,
            parallel: str = 'z'
        ) -> None:
        latex = "".join([c for c in latex if c not in r"$\{}"])
        self.name = name
        self.latex = latex 
        self.parent = parent
        if parent: self.dx, self.dy = parent.dx, parent.dy
        self.verbose = verbose 
        self.caching = caching
        child_kwargs = {'parent':parent, 'verbose':verbose, 'caching':caching}
        if len(components)==1 and type(path:=components[0])==str:
            components = (
                ScalarField(path+"/x", name=name+"_x_component", latex=f"${latex}_x$", **child_kwargs), 
                ScalarField(path+"/y", name=name+"_y_component", latex=f"${latex}_y$", **child_kwargs), 
                ScalarField(path+"/z", name=name+"_z_component", latex=f"${latex}_z$", **child_kwargs)
            )
        self.ndims = len(components)
        assert 1<self.ndims<4, f"Only 2-3 components are supported. {len(components)} were given"
        component_names = "xyz"
        self.components = []
        for name,val in zip(component_names, components): 
            comp = ScalarField(val, caching=self.caching) if type(val)==str else val
            self.components.append(comp)
            setattr(self, name, comp)
        self.set_parallel(parallel)

    def __len__(self) -> int: return len(self.x)
    def __abs__(self) -> np.ndarray:
        return np.array([
            np.sqrt(sum([
                c[i]**2 for c in self.components
            ])) for i in verbose_bar(range(len(self)), self.verbose, desc="taking magnitude...")
        ])
    def __getitem__(self, item: int|slice) -> np.ndarray:
        match type(item):
            case builtins.int: return np.array([c[item] for c in self.components])
            case builtins.slice:
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([
                    [
                        c[i] for c in self.components
                    ] for i in item_iters
                ])
    
    def dot(self, other) -> np.ndarray: 
        if isinstance(other, VectorField):
            assert self.ndims==3, "only 3D vector fields can be dotted at this time"
            assert other.ndims==3, "Can only dot into a 3D vector field at this time"
            if self.verbose: print("constructing A...")
            A = np.array([
                [
                    [
                        [self.x[k][i,j], self.y[k][i,j], self.z[k][i,j]]
                    for i in range(self.shape[0])]
                for j in range(self.shape[1])] 
            for k in verbose_bar(range(len(self)), self.verbose, desc="constructing A...")])
            B = np.array([
                [
                    [
                        [other.x[k][i,j], other.y[k][i,j], other.z[k][i,j]]
                    for i in range(other.shape[0])]
                for j in range(other.shape[1])] 
            for k in verbose_bar(range(len(other)), self.verbose, desc="constructing B...")])
            return np.sum(A * B, axis=2)
    def cross(self, other, k:int) -> np.ndarray:
        if type(other)==VectorField: 
            assert self.ndims==3, "only 3D vector fields can be crossed at this time"
            assert other.ndims==3, "Can only cross into a 3D vector field at this time"
            return np.array([
                self.y[k]*other.z[k] - self.z[k]*other.y[k], 
                self.z[k]*other.x[k] - self.x[k]*other.z[k],
                self.x[k]*other.y[k] - self.y[k]*other.x[k]
            ])

    def curlz(self, item: int|slice, order: int = 2):
        match type(item):
            case builtins.int: return curlz(self.x[item], self.y[item], order=order)
            case builtins.slice:
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([curlz(self.x[i], self.y[i], order=order) for i in item_iters])
    @cached_property
    def Jz(self): return np.array([np.nanstd(j) for j in self.curlz(slice(0,len(self)))])
    @cached_property
    def perp(self) -> np.ndarray: return np.array([
        np.sqrt(self.perpendicular[0][j]**2 + self.perpendicular[1][j]**2) 
        for j in verbose_bar(range(len(self)), self.verbose, desc="perpendicularizing")
    ])
    @cached_property
    def psi(self) -> np.ndarray: return np.array([
        calc_psi(Bx, By, self.dx, self.dy) for Bx, By in zip(self.x, self.y)
    ])       
    def set_parallel(self, component:str) -> None:
        match component.lower():
            case 'x':
                self.parallel = self.x 
                self.perpendicular = self.y, self.z 
            case 'y':
                self.parallel = self.y 
                self.perpendicular = self.x, self.z 
            case 'z':
                self.parallel = self.z 
                self.perpendicular = self.x, self.y 
    def movie(self, mode='mag', norm='none', cmap=default_cmap, **kwrg) -> None:
        match mode.lower():
            case 'mag'|'magnitude'|'abs':
                @show_video(name=self.name+"_magnitude", latex=f"$|{self.name}|$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return abs(self)
            case 'perp'|'perpendicular':
                @show_video(name=self.name+"_perp", latex=f"${self.name}_\perp$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return self.perp                
            case 'par'|'parallel':
                @show_video(name=self.name+"_par", latex=f"${self.name}_\parallel$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return np.array([self.parallel[i] for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, **kwrg)