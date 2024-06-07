import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
import inspect

@contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print
    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print

def progress_bar(iterator, **kwargs):
    with redirect_to_tqdm():
        for x in tqdm(iterator, **kwargs):
            yield x

def verbose_bar(iterator, verbose, **kwargs):
    return progress_bar(iterator, **kwargs) if verbose else iterator

def yesno(prompt: str):
    """
    prompt the user to either reply yes or no
    :param prompt: the yes/no question to be answered
    :return: True if yes False if no
    """
    response = input(prompt).lower()
    if 'y' in response and not 'n' in response:
        return True
    elif 'n' in response and not 'y' in response:
        return False
    else:
        def retry_yesno():
            retry_prompt = "Sorry I couldn't read that please respond with yes or no\n" + prompt
            retry_response = input(retry_prompt).lower()
            if 'y' in retry_response and not 'n' in retry_response:
                return True
            elif 'n' in retry_response and not 'y' in retry_response:
                return False
            else:
                raise ValueError("need a response with either y or n in it.")

        return retry_yesno()

# Spectrum Functions
def kspec(image: np.ndarray) -> np.ndarray:
    return np.absolute(np.fft.fftshift(np.fft.fft2(image) / (1. * image.shape[0] * image.shape[1])))

def kspec1d(image: np.ndarray, bins: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = kspec(image)
    Ny, Nx = image.shape
    kmag = np.hypot(*np.mgrid[
                     -Ny // 2: Ny // 2,
                     -Nx // 2: Nx // 2
                     ][::-1])
    kmin = np.nanmin(kmag[kmag != 0])
    kmax = np.nanmax(kmag)
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), bins + 1)
    kx = np.array([np.mean([kgrid[i], kgrid[i+1]]) for i in range(len(kgrid)-1)])
    ks = np.array([np.mean(k[np.where((kgrid[i] < kmag) & (kmag < kgrid[i+1]))]) for i in range(len(kgrid)-1)])
    kerr = np.array([np.std(in_bin:=k[np.where((kgrid[i] < kmag) & (kmag < kgrid[i+1]))])/np.sqrt(len(in_bin)) for i in range(len(kgrid)-1)])
    return kx, ks, kerr
