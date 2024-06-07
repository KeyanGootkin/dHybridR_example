#pysim imports
from pysim.parsing import File, Folder
from pysim.environment import frameDir, videoDir
#nonpysim imports
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage
from functools import wraps
import os

from matplotlib.colors import LinearSegmentedColormap
pink = "#E34F68"
lightpink = "#E39FAA"
blue = "#7350E6"
lightblue = "#AE9FE3"
shadow = "#1A1219"
manoaskies = LinearSegmentedColormap.from_list("manoaskies", [pink, blue])
manoaskies_centered = LinearSegmentedColormap.from_list("manoaskies_centered", [lightpink, pink, shadow, blue, lightblue])
default_cmap = plt.cm.plasma

# <||-----|-----|-----|-----|-----|-----|-----|-----|------|-----|-----|------|------|-----|-----|-----|-----|-----||>
#                                                   FUNCTIONS
# <||-----|-----|-----|-----|-----|-----|-----|-----|------|-----|-----|------|------|-----|-----|-----|-----|-----||>
# Ploting utils
def auto_norm(
    norm: str, 
    frames: np.ndarray, 
    linear_threshold: float|None = None, 
    center: float|None = None, 
    saturate: float|None = None
):
    frames = frames[(-np.inf < frames)&(frames < np.inf)]
    # set min/max IF saturate is None                          or IF saturate is a tuple                                          ELSE assume its a float
    low = np.nanmin(frames) if saturate is None else np.nanquantile(frames, 1-saturate[0]) if isinstance(saturate, tuple) else np.nanquantile(frames, 1-saturate)
    high = np.nanmax(frames) if saturate is None else np.nanquantile(frames, 0+saturate[1]) if isinstance(saturate, tuple) else np.nanquantile(frames, 0+saturate)
    match norm.lower():
        case "lognorm":
            if low < 0: raise ValueError(f"minimum is {low}, LogNorm only takes positive values")
            if low==0: low=np.nanmin(frames[frames!=0])
            return LogNorm(vmin=low, vmax=high)
        case "symlognorm":
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            if np.abs(mu)-sig > 0: raise TypeError("SymLogNorm is only designed for stuff close to zero!")
            return SymLogNorm(sig if linear_threshold is None else linear_threshold, vmin=low, vmax=high)
        case n if n in ["centerednorm", "twoslope", "twoslopenorm"]:
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            # for the center use center if give otherwise use 0 if mean is small, else use mean
            vcenter = center if not center is None else 0 if np.abs(mu)-sig > 0 else mu
            return TwoSlopeNorm(vmin=low, vcenter=vcenter, vmax=high)
        case _: return Normalize(vmin=low, vmax=high)

def tile(arr: np.ndarray) -> np.ndarray:
    return np.r_[np.c_[arr, arr, arr], np.c_[arr, arr, arr], np.c_[arr, arr, arr]]

# Plots
def show(
        field: np.ndarray,
        #should i tile the image?
        tile_image: bool = False,
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        figsize: tuple[float, float] = (10, 10),
        show: bool = True,
        #plot parameters
        cmap = default_cmap,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: list|None = None,
        #contour options
        contour: np.ndarray|None = None,
        contour_style: dict = {'levels': 10, 'colors': 'black'},
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #everything else goes into pcolormesh
        **kwargs
):
    # prep data
    assert (ndims:=len(field.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    image = tile(field) if tile_image else field
    # check axes
    if x is None: x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    else: assert (len(x), line(y)) == image.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    close_fig = True if fig is None else False
    show = show if fig is None else False
    if fig is None: (fig, ax) = plt.subplots(figsize=figsize)
    # plot data
    img = ax.pcolormesh(x, y, image, cmap=cmap, **kwargs)
    # colorbar
    if colorbar: 
        divider = make_axes_locatable(ax)
        colorbar_location = colorbar_style.pop("location") if "location" in colorbar_style.keys() else "right"
        cax = divider.append_axes(colorbar_location, **colorbar_style)
        fig.colorbar(img, cax=cax, ax=ax, ticks=cticks)
    # contour
    if not contour is None: ax.contour(contour, **contour_style)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    if close_fig: plt.close(fig)
    else: return fig, ax, img

def diagnose_frame(
        s,
        i: int,
        file_name: str,
        outdir: str = "./frames/diag/",
        track_params: list = [],
        full_path = False,
        return_plots: bool = False
):
    """
    make a diagnostic image for a simulation at some index i
    :param s: The Simulation object to pull data from
    :param i: The index at which to pull data
    :param outdir:
    :return:
    """
    mosaic: str = """
    ppbb
    ppbb
    eeee
    jjjj
    """
    # Fields
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(5, 5), constrained_layout=True)
    show(s.density[i], fig=fig, ax=axes['p'], norm=LogNorm(vmax=100, vmin=.1))
    axes['p'].set_title(r"$\rho$", fontsize=14)
    show(s.B.z[i], fig=fig, ax=axes['b'], vmax=5, vmin=0)
    axes['b'].set_title(r"$\vec{B}_z$", fontsize=14)
    # Energy Spectrum
    m = name2num(s.name)
    x = s.energy_grid[i]
    y = s.energy_grid[i] * s.energy_pdf[i]
    eline, = axes['e'].loglog(x, y, color='black')
    axes['e'].set_xlim(1e-1, 1e4)
    axes['e'].set_xlabel(r"$E$ [$V_A^2$]")
    axes['e'].set_ylim(1e-3, 1e2)
    axes['e'].set_ylabel(r"$Ef_E$ [$V_A^2$]")
    # Jz and other tracks
    for j in track_params: axes['j'].plot(s.tau, j, color="black")
    axes['j'].axvline(s.tau[i], color='black', ls='-.')
    jtrack = axes['j'].scatter(s.tau[i], j[i], color='red')
    axes['j'].set_xlabel(r"$\tau$")
    if len(track_params)==1: axes['j'].set_ylabel(r"$J_z$")
    fig.suptitle(rf"$\tau$ = {s.tau[i]:.2f}", fontsize=14)
    # Save figure
    if isinstance(full_path, str): 
        plt.savefig(full_path)
        return None if not return_plots else jtrack
    if not file_name.endswith(".png"): file_name += ".png"
    plt.savefig(outdir + f"/{s.name}/" + file_name)

# Videos
def video_plot(xs, ys, file, fps=10, compress=1, grid=True, scale='linear', **kwargs):
    fig, ax = plt.subplots(dpi=100)
    xplot, yplot = nan_clip(xs[0], ys[0])
    yplot[yplot == 0] = 1e-9
    line, = ax.plot(xplot, yplot, **kwargs)
    ax.set_ylim(-6, np.nanmax(ys))
    ax.set_xlim(-1, 3)
    if grid: ax.grid()
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    def update(t):
        index = int(t * fps * compress)
        if index < len(ys) - 1:
            xplot, yplot = nan_clip(xs[index], ys[index])
            yplot[yplot == 0] = 1e-9
            line.set_xdata(xplot)
            line.set_ydata(yplot)
            return mplfig_to_npimage(fig)
        xplot, yplot = nan_clip(xs[-1], ys[-1])
        line.set_xdata(xplot)
        line.set_ydata(yplot)
        return mplfig_to_npimage(fig)

    animation = VideoClip(update, duration=len(ys) / compress / fps)
    animation.write_videofile(file, fps=fps, logger=None, progress_bar=False)

def make_video(name: str, frames: str = 'frames', fps: int = 12, outdir='.', verbose=False):
    """
    Takes images from directory "frames" and makes it into a video
    :param name: name of video, file will be outdir+"/"+name+".png"
    :param frames: the directory where the frames are stored
    :param fps: frames per second
    :param outdir: which directory to put the video in
    """
    image_folder = frames
    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files = list(np.array(image_files)[np.argsort(image_files)])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{outdir}/{name}.mp4', verbose=verbose)

def monitor_video(s, outdir="./monitor/"):
    ensure_path(monitor_frames:=f"/home/x-kgootkin/turbulence/frames/monitor/{s.name}/")
    already_there = glob(monitor_frames+"*")
    jz = s.Jz()
    print(f"Monitor Activation: {dtime.now()}")
    for i in progress_bar(range(len(jz))):
        full_path = f"{monitor_frames}monitor_{str(i).zfill(6)}.png"
        if os.path.exists(full_path): continue
        diagnose_frame(s, i, '', full_path=full_path, track_params=[jz])
    make_video(f"{s.name}_monitor", monitor_frames, outdir=outdir)

# <||-----|-----|-----|-----|-----|-----|-----|-----|------|-----|-----|------|------|-----|-----|-----|-----|-----||>
#                                                   DECORATORS
# <||-----|-----|-----|-----|-----|-----|-----|-----|------|-----|-----|------|------|-----|-----|-----|-----|-----||>

def line_video(
    compress: int = 1, fps: int = 10, fs: tuple = (5,5),grid: bool = True,
    xlimits = None, ylimits = None, scale = None, xscale = None, yscale = None,
    yticks = None, yticklabels = None, hlines=[], vlines=[], **video_kwargs
):
    def line_video_decorator(func):
        @wraps(func)
        def line_video_wrapper(*args, save="default", **kwargs):
            # Calculate data via func
            xs, ys = func(*args, **kwargs)
            # Setup plot
            fig, ax = plt.subplots(figsize=fs)
            xplot, yplot = nan_clip(xs[0], ys[0])
            yplot[yplot == 0] = 1e-9
            # Plot first line
            line, = ax.plot(xplot, yplot, **video_kwargs)
            # Set up axes
            for hl in hlines: ax.axhline(hl, color=video_kwargs["color"] if "color" in video_kwargs.keys() else "black")
            for vl in vlines: ax.axvline(vl, color=video_kwargs["color"] if "color" in video_kwargs.keys() else "black")
            ax.set_xlim(
                np.nanmin(xs) if xlimits is None else xlimits[0], 
                np.nanmax(xs) if xlimits is None else xlimits[1]
            )
            ax.set_ylim(
                np.nanmin(ys) if ylimits is None else ylimits[0],
                np.nanmax(ys) if ylimits is None else ylimits[1]
            )
            if not yticks is None: ax.set_yticks(yticks)
            if not yticklabels is None: ax.set_yticklabels(yticklabels)
            if grid: ax.grid()
            if not scale is None:
                ax.set_xscale(scale)
                ax.set_yscale(scale)
            else: 
                if not xscale is None: ax.set_xscale(xscale)
                if not yscale is None: ax.set_yscale(yscale)
            # define video making function
            def update(t):
                index = int(t * fps * compress)
                if index < len(ys) - 1:
                    xplot, yplot = nan_clip(xs[index], ys[index])
                    yplot[yplot == 0] = 1e-9
                    line.set_xdata(xplot)
                    line.set_ydata(yplot)
                    return mplfig_to_npimage(fig)

                xplot, yplot = nan_clip(xs[-1], ys[-1])
                line.set_xdata(xplot)
                line.set_ydata(yplot)
                return mplfig_to_npimage(fig)
            # save video
            animation = VideoClip(update, duration=len(ys) / compress / fps)
            animation.write_videofile(save+".mp4", fps=fps, logger=None)
        return line_video_wrapper
    return line_video_decorator

def show_video(
    name: str = 'none', 
    latex: str = None, 
    cmap = default_cmap, 
    norm="none", 
    figsize=(5,5)
):
    def simple_video_decorator(func):
        @wraps(func)
        def simple_video_wrapper(
            s, *args, 
            cmap=cmap, norm=norm, figsize=figsize, 
            savedir=videoDir.path, compress=1, fps=10, **kwargs
        ):
            # make a directory to store frames in
            Folder(
                fpath := f"{frameDir.path}/{s.name}/{name}/"
            ).make()
            Folder(savedir).make()
            # make the frames and return as a numpy array
            frames = func(s, *args, **kwargs)
            # plot the frames
            fig,ax = plt.subplots(figsize=figsize)
            normalization = norm if not isinstance(norm, str) else auto_norm(norm, frames)
            *_,img = show(
                frames[0], 
                fig = fig,
                ax = ax,
                cmap=cmap, 
                norm=normalization, 
                title=f"{latex}" if isinstance(latex, str) else f"{name}"
            )
            def update(t):
                index = int(t * fps * compress)
                if index < len(frames) - 1:
                    img.set_array(frames[index])
                    return mplfig_to_npimage(fig)
                img.set_array(frames[-1])
                return mplfig_to_npimage(fig)
            animation = VideoClip(update, duration=len(frames) / compress / fps)
            animation.write_videofile(f"{savedir}/{s.name}_{name}.mp4", fps=fps)
        return simple_video_wrapper
    return simple_video_decorator
