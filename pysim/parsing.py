#pysim imports
from pysim.utils import yesno
#nonpysim imports
from glob import glob 
from shutil import copy, move, copytree, rmtree
from os.path import isdir, isfile, exists
from os import mkdir, remove
from functools import cached_property


def ensure_path(path):
    parts = path.strip().split('/')
    for i in range(len(parts)):
        if "/".join(parts[:i]) in "/home/x-kgootkin/": continue
        if not exists("/".join(parts[:i])):
            mkdir("/".join(parts[:i]))

class Folder:
    def __init__(self, path:str, master=None) -> None:
        self.path = path.replace("\\", "/")
        self.name = self.path.split("/")[-1] if len(self.path.split('/')[-1])>0 else self.path.split("/")[-2]
        self.master = master if not isinstance(master, str) else Folder(master)

    def exists(self) -> bool: return exists(self.path)
    @cached_property
    def children(self) -> list: return glob(self.path+"/*")
    def ls(self) -> None: print("\n".join(self.children))
    def make(self) -> None: ensure_path(self.path)
    def copy(self, destination:str) -> None: copytree(self.path, destination)
    def update(self) -> None:
        assert self.master, "No master copy to update from."
        if self.exists(): self.delete(interactive=False)
        self.master.copy(self.path)
        self = Folder(self.path, master=self.master)
    def delete(self, interactive=True) -> None:
        if interactive and not yesno(f"Are you sure you want to permanently delete {self.path} and all of its contents?\n"): 
            return None
        rmtree(self.path)

class File:
    def __init__(self, path:str, master=None, executable:bool=False) -> None:
        self.path = path.replace("\\", "/")
        self.parent = "/".join(self.path.split("/")[:-1])
        self.name = self.path.split("/")[-1]
        self.extension = self.name.split(".")[-1] if "." in self.name else None
        self.master = master if not isinstance(master, str) else File(master)
        self.executable = executable

    def copy(self, destination:str): copy(self.path, destination)
    def move(self, destination:str): 
        move(self.path, destination)
        self = File(destination, master=self.master)
    def exists(self): return exists(self.path)
    def update(self) -> None:
        print('updating')
        assert self.master, "No master copy to update from."
        if self.exists(): self.delete(interactive=False)
        self.master.copy(self.path)
        self = File(self.path, master=self.master)
    def delete(self, interactive=True) -> None:
        if interactive and not yesno(f"Are you sure you want to permanently delete {self.path} and all of its contents?\n"): 
            return None
        remove(self.path)        