# -*- coding: utf-8 -*-
"""
Export functionalities for figures and plots from python and Matplotlib

Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""

import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def predefined_dims() -> dict:

    # set height and width for each style
    dims = dict()

    # use a dict for easier handling (instead of many if statements)
    # dims[key] = [width, height] in cm
    #TODO maybe find prettier way to handle offset
    dims['presentation_1x1'] = [29.21+0.254, 13.5636+0.254]         # fill a full 16:9 slide body
    dims['presentation_1x2'] = [29.21/2.0+0.254, 13.5636+0.254]     # 2 figures (1 x 2) per slide
    dims['presentation_1x3'] = [29.21/3.0+0.254, 13.5636+0.254]     # 3 figures (1 x 3) per slide
    dims['presentation_2x2'] = [29.21/2.0+0.254, 13.5636/2+0.254]   # 4 figures (2 x 2) per slide
    dims['presentation_2x3'] = [29.21/3.0+0.254, 13.5636/2+0.254]   # 6 figures (2 x 3) per slide
    dims['presentation_2x1'] = [29.21+0.254, 13.5636/2+0.254]       # 2 figures (2 x 1) per slide

    #dims['document_1x1'] = [28, 20]         # square figure spanning text width
    #dims['document_1x1_wide'] = [28, 10]    # wide figure spanning text width
    #dims['document_1x2'] = [12, 10]         # 2 figures on (1 x 2) grid
    #dims['document_2x2'] = [6, 6]           # 4 figures on (2 x 2) grid
    #dims['document_1x3'] = [5, 6]           # 6 figures on (2 x 3) grid
    #dims['document_2x3'] = [5, 5]           # 6 figures on (2 x 3) grid
   
    return dims


def export_figure(fig,
                  name: str,
                  savedir: str = None,
                  style: str = None,
                  width: float = None,
                  height: float = None,
                  resolution: int = 300) -> None:
    """ Export to external file 

    fig:        plt figure object [fig, ax = plt.subplots()]
    name:       file name, including file ending
    savedir:    directory where to save the image file. Default: current location

    style: pre-defined export styles for quick access:
        - presentation-1/1
        - presentation-1/2_wide / _tall
        - presentation-1/3_wide / _tall
        - presentation-1/4_wide / _tall

    width:  width in cm, optional. Will over-write the style argument
    height: heigth in cm, optional. Will over-write the style argument

    """

    # check if directory is valid
    if savedir is None:
        # case 1: user does not specify directory
        savedir = os.getcwd()                               # choose current directory to save the figure
    
    elif not os.path.isdir(savedir):
        # case 2: user specifies directory 
        raise ValueError('save directory is not valid!')    # raise Error if specified directory is not valid

    # load predefined style configurations
    dims = predefined_dims()

    # get the aspect ratio of the figures axis'
    aspect_ratio = fig.gca().get_aspect()

    # check which width and height is specified
    if (width is not None) and (height is not None):
        # case 1: user specifies width and height
        width_ = width
        height_ = width

        if aspect_ratio == 1.0:
            # favor width argument, when aspect ratio is equal
            width_ = width_
            height_ = None

    elif (style is not None) and (style in dims.keys()):
        # case 2: user specifies predefined style configuration
        width_ = dims[style][0]
        height_ = dims[style][1]

        if aspect_ratio == 1.0:
            # favor width argument, when aspect ratio is equal
            width_ = width_
            height_ = None

    elif not (style in dims.keys()):
        # case 3: user does not specify width and height or style from dict
        raise ValueError(f'Please choose one style out of {dims.keys()}')

    # change figure size
    fig.set_size_inches(width_*0.393701, height_*0.393701)  # this is in inches only
                                                             #convert cm to inch by * 0.3937007874

    # detect file ending
    file_ending = name.split('.')[-1]

    # compose the save string
    dir_str = os.path.join(savedir, name)

    print('exporting figure ...')
    print(f'\t style={style}')
    print(f'\t directory={savedir}')
    print(f'\t filename={name}')
    print(f'\t fileending={file_ending}')

    # export figure dependent on file ending
    if file_ending == 'tikz':
        # case 1: export tikz file (WIP)
        import tikzplotlib
        tikzplotlib.clean_figure()          # get rid of all non-visible data points
        tikzplotlib.save(filepath=dir_str)

    else:
        # case 2: figure export as impage and png image of certain size and resolution
        # bbox_inches takes care of keeping everything inside the frame that is being exported
        plt.savefig(fname=dir_str,
                    dpi=resolution,
                    bbox_inches="tight",
                    format=file_ending,
                    )


if __name__ == '__main__':

    mpl.style.reload_library()
    with plt.style.context('cps_presentation'):

        data = np.random.randn(10)
        fig, ax = plt.subplots()
        ax.plot(data, data)
        ax.plot(data, -data)
        plt.xlabel('xlabel')
        plt.ylabel('ylabel')
        plt.legend(['some data', 'different data'])
        fig.patch.set_facecolor('gray')

    styles = predefined_dims()

# export figures to according file
    for style in styles:
        export_figure(fig, name='test_' + style + '.png',
                      savedir='export_samples', style=style)
        export_figure(fig, name='test_' + style + '.pdf',
                      savedir='export_samples', style=style)
        export_figure(fig, name='test_' + style + '.svg',
                      savedir='export_samples', style=style)
        export_figure(fig, name='test_' + style + '.tikz', 
                      savedir='issue09/export_samples', style=style)

    plt.show()
