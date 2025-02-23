
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, sparse
from math import ceil, floor
import os
import pickle
#import mihailpkg.data_plotter as dp
import tkinter.filedialog as fd
import tkinter as tk
import struct

def get_shortcut_absolute_path(path):
    target = ''

    with open(path, 'rb') as stream:
        content = stream.read()
        # skip first 20 bytes (HeaderSize and LinkCLSID)
        # read the LinkFlags structure (4 bytes)
        lflags = struct.unpack('I', content[0x14:0x18])[0]
        position = 0x18
        # if the HasLinkTargetIDList bit is set then skip the stored IDList
        # structure and header
        if (lflags & 0x01) == 1:
            position = struct.unpack('H', content[0x4C:0x4E])[0] + 0x4E
        last_pos = position
        position += 0x04
        # get how long the file information is (LinkInfoSize)
        length = struct.unpack('I', content[last_pos:position])[0]
        # skip 12 bytes (LinkInfoHeaderSize, LinkInfoFlags, and VolumeIDOffset)
        position += 0x0C
        # go to the LocalBasePath position
        lbpos = struct.unpack('I', content[position:position+0x04])[0]
        position = last_pos + lbpos
        # read the string at the given position of the determined length
        size= (length + last_pos) - position - 0x02
        temp = struct.unpack('c' * size, content[position:position+size])
        target = ''.join([chr(ord(a)) for a in temp])
    return target

def baseline_als(y, lam=1e7, p=0.01, niter=10):
    '''


    Parameters
    ----------
    y : ndarray
        Spectrum.
    lam : float, optional
        DESCRIPTION. The default is 1e7.
    p : float, optional
        DESCRIPTION. The default is 0.01.
    niter : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    z : ndarray
        Background.

    '''
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
      W = sparse.spdiags(w, 0, L, L)
      Z = W + lam * D.dot(D.transpose())
      z = sparse.linalg.spsolve(Z, w*y)
      w = p * (y > z) + (1-p) * (y < z)
    return z

def redistribute(x,y,x_new):
    '''


    Parameters
    ----------
    x : ndarray
        DESCRIPTION.
    y : ndarray
        DESCRIPTION.
    x_new : ndarray
        DESCRIPTION.

    Returns
    -------
    ndarray
        y_new.

    '''
    def interpolate(x0):
        i = np.searchsorted(x, x0, side="left")

        if i == 0:
            return y[i]
        if i == y.size:
            return y[i-1]
        if x[i] == x0:
            return y[i]

        if x[i] > x0:
            x1,x2 = x[i-1], x[i]
            y1,y2 = y[i-1], y[i]
            y_new = (y2-y1)/(x2-x1) * (x0-x1) + y1
            return y_new

    # Ensure that the array is sorted in ascending order
    if np.all(x[:-1] <= x[1:]): #ascending order
        pass
    elif np.all(x[:-1] >= x[1:]): #descending order
        x, x_new = -x, -x_new
    else:
        print('array is not sorted')

    y_new = np.zeros(y.size)
    for i in range(y.size):
        y_new[i] = interpolate(x_new[i])

    return y_new


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def fit_poly(x, y, deg, stats=False):
    A = np.zeros((x.size,deg+1))
    for col in range(deg+1):
        A[:,col] = x**(deg - col)
    x, res, rank, s = np.linalg.lstsq(A, y,rcond=None)
    if not stats:
        return x
    if stats:
        return x, res

def delim_spectrum(filename, skiplines = 0):
    with open(filename, 'r') as file:
        all_lines = file.readlines()

    lines = all_lines[skiplines:]
    arr = np.zeros((len(lines),2))
    for i,line in enumerate(lines):
        for j,e in enumerate(line.strip().split(' ')):
            if e == '': continue
            if j==0: arr[i,0] = float(e)
            else: arr[i,1] = float(e)


    header = ''
    for line in all_lines[:25]: header += line + '\n'
    np.savetxt(filename, arr, delimiter='\t', header=header, fmt='%.4e')

def average(arr, n):
    group_size = n
    number_of_groups = arr.size//group_size
    averaged_arr = np.zeros(number_of_groups)
    for i in range(number_of_groups):
        averaged_arr[i] = np.average(arr[i*group_size:(i+1)*group_size])
    return averaged_arr

def run(file, path):
    '''
    Execute a python script in its directory

    Parameters
    ----------
    file : str
        Name of python script.
    path : str
        Directory containing the script.

    Returns
    -------
    None.

    '''
    def run_local(file, path):
        os.chdir(path)
        exec(open(file).read())
    CWD = os.getcwd()
    run_local(file, path)
    os.chdir(CWD)

def parse_file(filename, delim='\t'):
    """Parse the data file for numeric data and return the array."""

    with open(filename) as file:
        text_file = file.readlines()
    numeric_data = []

    for i in range(1,100):
        try:
            last_row = text_file[-1*i].strip().split(delim)
            for element in last_row:
                float(element)
            num_col = len(last_row)
            if num_col < 2:
                continue
            break
        except:
            continue

    for i,row in enumerate(text_file):
        columns_list = row.strip().split(delim)
        for j,element in enumerate(columns_list):
            try:                                            #if the row contains the correct number of elements and these are
                numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
                columns_list[j] = numeric_element
                if j == num_col - 1:
                    numeric_data.append(columns_list)
            except: continue
    data_array = np.array(numeric_data)
    return data_array

def select_files():
    root = tk.Tk()
    root.update()
    filenameslist = fd.askopenfilenames(filetypes = [('All files','*.*'),
                                                  ('Text files','*.txt'),
                                                  ('CSV files','*.csv'),
                                                  ('FuelCell3 files','*.fcd'),
                                                  ('NPY file','*.npy'),
                                                  ('MLG file','*.mlg')])
    root.destroy()
    data = {}
    for filename in filenameslist:
        arr = parse_file(filename)
        data[filename] = arr
    return data

def mass_to_charge_cont(mass, charge, npts_to_avg = 50):
    '''


    Parameters
    ----------
    mass : ndarray
        Array of mass values from experiment.
    charge : ndarray
        Array of charge values from experiment.
    npts_to_avg : int, optional
        The number of points used in the moving difference. The default is 50.

    Returns
    -------
    m2c_cont : ndarray
        Differential mass to charge ratio based on a moving difference over the
        mass and charge arrays.
    slice_name : slice
        Since a moving difference requires a truncated array, use this slice
        object to slice data for plotting the mass to charge ratio

    '''
    charge = resize_array(charge, mass.size) #resize charge array to be compatable with mass array

    diff_mass_cont = np.zeros(mass.size - npts_to_avg) #differential mass using moving difference
    diff_charge_cont = np.zeros(mass.size - npts_to_avg) #differential charge using moving difference
    for i in range(mass.size - npts_to_avg):
        diff_mass_cont[i] = mass[i + npts_to_avg] - mass[i]
        diff_charge_cont[i] = charge[i + npts_to_avg] - charge[i]

    m2c_cont = -diff_mass_cont * 96485 / diff_charge_cont
    slice_name = slice(round(npts_to_avg/2),-round(npts_to_avg/2))
    return m2c_cont, slice_name


def pac(L, LAMBDA, RG):
    '''
    Model ET kinetics dependant PAC feedback current.

    For details see doi: https://doi.org/10.1002/cphc.200900600

    Parameters
    ----------
    L : ndarray
        Tip-substrate distance normalized by electrode radius.
    LAMBDA : float
        ET rate constant normalised by D/a.
    RG : float
        Tip radius normalized by electrode radius.

    Returns
    -------
    ndarray
        Tip current normalized by current at infinite distance.

    '''
    def Ni_cond(L, RG):
        def alpha(RG):
            return (np.log(2)
                    + np.log(2) * (1 - 2/np.pi * np.arccos(1/RG))
                    - np.log(2) * (1 - (2/np.pi * np.arccos(1/RG))**2))
        def beta(RG):
            return (1
                    + 0.639 * (1 - 2/np.pi * np.arccos(1/RG))
                    - 0.186 * (1 - (2/np.pi * np.arccos(1/RG))**2))
        return (alpha(RG)
                + np.pi / (4*beta(RG)*np.arctan(L))
                + (1 - alpha(RG) - 1/(2*beta(RG))) * (2/np.pi*np.arctan(L)))

    def Ni_ins(L, RG):
        return ((2.08/RG**0.358 * (L - 0.145/RG) + 1.585)
                / (2.08/RG**0.358 * (L + 0.0023*RG) + 1.57 + np.log(RG)/L + 2/(np.pi*RG) * np.log(1 + np.pi*RG/(2*L))))

    return (Ni_cond((L + 1/LAMBDA), RG)
            + (Ni_ins(L,RG) - 1)
            / ((1 + 2.47*RG**0.31*L*LAMBDA)*(1 + L**(0.006*RG + 0.113)*LAMBDA**(-0.0236*RG + 0.91))))

def linear_interpolation(y1, y2, x1, x2, x):
    m = (y2 - y1) / (x2 - x1)
    y = m * (x - x1) + y1
    return y

def fft_low_pass(array,cutoff_freq,baseline = 0, visualize = False):
    yf = fft.rfft(array) # discrete Fourier Transform
    xf = fft.fftfreq(array.size) # sample frequencies
    cut_f_signal = yf.copy()
    cut_f_signal[index_of_value(cutoff_freq,xf):] = baseline * array.size / 2 # remove frequencies above cutoff_freq
    cut_signal = fft.irfft(cut_f_signal)
    if visualize:
        fig,ax = plt.subplots(ncols=2, figsize=(8,4))
        ax[0].plot(array, label='Original signal')
        ax[0].plot(cut_signal, label = 'Processed signal')
        ax[0].set_xlabel('array index / i')
        ax[0].legend()
        ax[1].plot(2/array.size*np.abs(yf), label='FFT Original')
        ax[1].plot(2/array.size*np.abs(cut_f_signal), label='FFT Processes')
        ax[1].set_xlabel('sampling frequency / i$^{-1}$')
        ax[0].legend()
        return cut_signal
    return cut_signal

def index_of_value(value,array):
    return np.argmin(np.abs(array - value))

def dict_value_from_num(dict,n):
    return dict[list(dict.keys())[n]]

def resize_array(arr,newsize):
    if arr.size == newsize: return arr
    def arr_continuous(x):
        if x.is_integer():
            return arr[int(x)]
        else:
            x2, x1 = ceil(x), floor(x)
            y2, y1 = arr[x2], arr[x1]
            m = (y2 - y1) / (x2 - x1)
            y = m*(x - x1) + y1
            return y

    arr_resized = np.zeros(newsize)
    for i in range(newsize):
        x = i*(arr.size - 1)/(newsize - 1)
        arr_resized[i] = arr_continuous(x)
    return arr_resized

def text_figure(desc, fig=None, height=0.2, fontsize=10):
    if fig and height < 1:
        axes = fig.get_axes()
        for ax in axes:
            ax.axis('off')
    if not fig:
        fig,ax = plt.subplots(figsize = (8,6))
        ax.axis('off')
    fig.text(0,height,desc,fontsize = fontsize, family = 'consolas')
    fig.tight_layout()
    return fig

def generate_figures(n):
    fig_refs = []
    ax_refs = []
    for i in range(n):
        fig,ax = plt.subplots()
        fig_refs.append(fig)
        ax_refs.append(ax)
    return fig_refs,ax_refs

def load_and_plot_files():
    """Use the data plotter to load data files into a file dictionary."""
    dp.main()
    print('-'*20)
    for i,key in enumerate(dp.data):
        print(i,'\t',key)
    print('-'*20)
    data = dp.data.copy()
    dp.data = {'label':{}}
    dp.label = dp.data['label']
    return data

def convert_all_txt_files(delimiter = '\t'):
    data = {'label':{}, 'metadata':{}}
    label = data['label']
    metadata = data['metadata']

    def parse_file(filename):
        """Parse the data file for numeric data and append data array to the dictionary."""

        with open(filename) as file:
            text_file = file.readlines()
        filename = filename[:-4]
        numeric_data = []
        delim = delimiter

        for i in range(1,100):
            try:
                last_row = text_file[-1*i].strip().split(delim)
                for element in last_row:
                    float(element)
                num_col = len(last_row)
                if num_col < 2:
                    continue
                break
            except:
                continue

        reading_data = False
        for i,row in enumerate(text_file):
            if row[0] == '#': continue
            if not reading_data: metadata_row = i
            columns_list = row.strip().split(delim)
            for j,element in enumerate(columns_list):
                try:                                            #if the row contains the correct number of elements and these are
                    numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
                    columns_list[j] = numeric_element
                    reading_data = True
                    if j == num_col - 1:
                        numeric_data.append(columns_list)
                except:
                    if len(columns_list) == num_col and j == 0:
                        label[filename] = columns_list          #it must be a label describing the data
                    continue
        data_array = np.array(numeric_data)
        data[filename] = data_array                         #add the data to the dictionary
        if filename not in label:
            label[filename] = list(range(num_col))
        if filename not in metadata:
            metedata_text = ''
            for row in text_file[:metadata_row]: metedata_text += row
            metadata[filename] = metedata_text

    for filename in os.listdir():
        if filename[-3:] == 'txt':
            try:
                parse_file(filename)
            except:
                continue
    with open('preprocessed data.pickle','wb') as file:
        pickle.dump(data, file)

def secm_1Dto2D(nx3_array):

    data_array = nx3_array
    x = data_array[:,0]
    y = data_array[:,1]
    z = data_array[:,2]
    xperiod = np.argmax(x)
    xmax = x[xperiod]
    yperiod = np.argmax(y)
    ymax = y[yperiod]
    if xperiod < yperiod: #longaxis = 'y'
        xstep = x[1] - x[0]
        ystep = y[xperiod + 1] - y[0]

    if xperiod > yperiod: #longaxis = 'x'
        ystep = y[1] - y[0]
        xstep = x[yperiod + 1] - x[0]

    xpts = int(xmax/xstep + 1)
    xaxis = np.linspace(x[0],xmax,xpts)
    ypts = int(ymax/ystep + 1)
    yaxis = np.linspace(y[0],ymax,ypts)
    X,Y = np.meshgrid(xaxis,yaxis)
    Z = np.empty(xpts * ypts)
    Z[:] = np.nan
    Z[:z.size] = z
    Z = Z.reshape(ypts,xpts)
    return X,Y,Z

def plot_secm(flat_array, normalization_current = 1, current_limits=(None,None),
              number_of_ticks = 6, label='', label_loc = (0.035,0.88),
              imshow_kwargs_dict = {},
              colorbar_kwargs_dict = {}):

    '''

    Parameters
    ----------
    flat_array : ndarray
        A nx3 array of x,y,i values from SECM experiment.
    normalization_current : float, optional
        Bulk steady state current. The default is 1.
    label : str, optional
        Label to place on the figure. The default is ''.
    label_loc : tuple, optional
        Location of label in axes units. The default is (0.035,0.88).
    imshow_kwargs_dict : dict, optional
        Keyword arguments to send to imshow function.
    colorbar_kwargs_dict : dict, optional
        Keyword arguments to send to colorbar function.

    Returns
    -------
    fig : Figure
        The figure object.
    ax : Axes
        The axes object.

    '''
    default_imshow_kwargs_dict = {'interpolation':'bicubic','cmap':'Spectral_r'}
    default_colorbar_kwargs_dict = {'orientation':'vertical','pad':0.01,'fraction':0.05,'aspect':40,'label':'Normalized current ($\mathit{i}_{p}\ /\ \mathit{i}_{p,lim}$)'}


    if (current_limits[0] != None) and (current_limits[1] != None):
        default_imshow_kwargs_dict['vmin'] = current_limits[0]
        default_imshow_kwargs_dict['vmax'] = current_limits[1]
        default_colorbar_kwargs_dict['ticks' ] = list(np.linspace(current_limits[0],current_limits[1],number_of_ticks))

    imshow_kwargs_dict = {**default_imshow_kwargs_dict, **imshow_kwargs_dict}
    colorbar_kwargs_dict = { **default_colorbar_kwargs_dict, **colorbar_kwargs_dict }

    fig,ax = plt.subplots()
    X,Y,Z = secm_1Dto2D(flat_array)
    Z = Z / normalization_current
    CS = ax.imshow(Z, extent=[X.min(),X.max(),Y.min(),Y.max()], origin='lower',**imshow_kwargs_dict)
    ax.set_aspect('equal')
    ax.text(*label_loc, label, transform=ax.transAxes)
    cbar = fig.colorbar(CS,ax = ax,**colorbar_kwargs_dict)
    return fig,ax

def array_from_number(data,n,alias = False):
    keys = [*data]
    if alias:
        array = data[keys[n]]
    if not alias:
        array = np.copy(data[keys[n]])
    return array


# def dataFrame_from_numbers(identifier,file_dictionary):
#     """
#     Take a dictionary of data Array objects loaded from files and generate a dataFrame object(s).

#     A file has a 'filename' and an int as identifiers.
#         The 'filename' is its name in the directroy from which the file is loaded
#         The int is the order in which the file is loaded
#     A file consists of a data_array and a column_label_list
#         The data_array is a float Array of experimental data
#         The column_label_list describes each column of data in the Array

#     file_dictionary = {'label':{'filename1':[col1_label,col2_label,...],...},
#                        'filename1':[data_col1,data_col2,...],
#                        'filename2':[data_col1,data_col2,...],
#                        ...}

#     Parameters
#     ----------
#     identifier : int, list of ints, or 'all'
#         Identifies which file(s) to convert into DataFrame object(s) based on
#         the file's position in the data dictionary, starting from 1.
#     file_dictionary : dict
#         Dictionary that stores the loaded data using 'filename':data_array
#         'key':value pairs. The 0th entry is a dictionary of 'filename':data_label
#         that stores the labels of each column for a given file

#     Returns
#     -------
#     A DataFrame object representing the file data, or a list of DataFrame objects
#     """

#     pd.DataFrame.filename = ''
#     ailias_repr = pd.DataFrame.__repr__
#     def modified_repr(self):
#         return self.filename + '\n' + ailias_repr(self)
#     pd.DataFrame.__repr__ = modified_repr

#     label_dictionary = file_dictionary['label']
#     filename_list = list(file_dictionary.keys())

#     if type(identifier) == int:
#         key = filename_list[identifier]
#         dataFrame = pd.DataFrame(data= file_dictionary[key],
#                                  columns=label_dictionary[key],
#                                  copy=True)
#         dataFrame.filename = key
#         return dataFrame

#     if type(identifier) == list:
#         dataFrame_list = []
#         for idnum in identifier:
#             key = filename_list[idnum]
#             dataFrame = pd.DataFrame(data= file_dictionary[key],
#                                      columns=label_dictionary[key],
#                                      copy=True)
#             dataFrame.filename = key
#             dataFrame_list.append(dataFrame)
#         return dataFrame_list

#     if type(identifier) == str:
#         if identifier == 'all':
#             dataFrame_list = []
#             for idnum in range(1,len(filename_list)):
#                 key = filename_list[idnum]
#                 dataFrame = pd.DataFrame(data= file_dictionary[key],
#                                          columns=label_dictionary[key],
#                                          copy=True)
#                 dataFrame.filename = key
#                 dataFrame_list.append(dataFrame)
#             return dataFrame_list

#         else:
#             dataFrame_list = []
#             for filename in filename_list:
#                 if identifier in filename:
#                     dataFrame = pd.DataFrame(data= file_dictionary[filename],
#                                              columns=label_dictionary[filename],
#                                              copy=True)
#                     dataFrame.filename = key
#                     dataFrame_list.append(dataFrame)
#             return dataFrame_list




def antideriv(x,y):
    """Take 2 Arrays x and y = f(x) and return Array z = F(x) = int(x0,x,y)."""
    N = y.size
    z = np.zeros(N)
    for i in range(1,N):
        z[i] = z[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return z

def differentiate(y ,x=None, npts=1):
    """Differentiate y array using n point moving difference"""
    dydx = np.zeros(y.size - npts)
    if type(x) == type(None):
        for i in range(dydx.size):
            dydx[i] = y[i + npts] - y[i]
        return dydx
    elif x.size == y.size:
        for i in range(dydx.size):
            dy = y[i + npts] - y[i]
            dx = x[i + npts] - x[i]
            dydx[i] = dy/dx
        return dydx, x[npts//2:dydx.size + npts//2]
    else:
        print('Error, x and y arrays must be same size')

def copy_labels(fig_or_ax,source_fig_or_ax):
    """Copy axes labels from one figure to another."""
    if 'Axes' in str(type(fig_or_ax)):
        fig_or_ax.set_xlabel(source_fig_or_ax.get_xlabel())
        fig_or_ax.set_ylabel(source_fig_or_ax.get_ylabel())

    if 'Figure' in str(type(fig_or_ax)):
        ax0 = fig_or_ax.get_axes()[0]
        ax1 = source_fig_or_ax.get_axes()[0]

        ax0.set_xlabel(ax1.get_xlabel())
        ax0.set_ylabel(ax1.get_ylabel())

def fig_to_data(fig,filename):
    """
    Create a .mlg data file from a figure.

    Takes a figure with multiple lines plotted on a single axes and generates
    a multi line graph (.mlg) file. The file structure is

    X units    Y units
    line0 label
    x0    y0
    x1    y1
    x2    y2
    ...
    line1 label
    x0    y0
    x1    y1
    x2    y2


    Parameters
    ----------
    fig : Figure
        Figure with one axes containing data lines
    filename : string
        name of file created, no extension

    Returns
    -------
    None.

    """
    with open(filename + '.mlg','w') as file:
        ax = fig.get_axes()[0]
        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()
        file.write(x_label + '\t' + y_label + '\n')
        for line in ax.lines:
            label = line.get_label()
            file.write(label + '\n')
            xdata,ydata = line.get_data()
            for i in range(xdata.size):
                file.write(str(xdata[i]) + '\t' + str(ydata[i]) + '\n')

    fig.savefig(filename + '.png')

def data_to_fig(filename, fig=None, ax=None):
    """Open a multiple line graph file as a figure."""
    with open(filename,'r') as file:
        textfile = file.readlines()
    if fig==None or ax==None:
        fig,ax = plt.subplots()
    ax.set_xlabel(textfile[0].split('\t')[0].strip())
    ax.set_ylabel(textfile[0].split('\t')[1].strip())
    label = textfile[1].strip()

    N=2 #tracks index of start of line data sequence and used to
        #incriment outer (row) loop when end of data sequence is reached
    for i,row in enumerate(textfile[2:]):
        element_list = row.split('\t')

        for j,element in enumerate(element_list):

            try:
                numeric_element = float(element)
                element_list[j] = numeric_element

            except ValueError:

                data = np.array(textfile[N:i])
                ax.plot(data[:,0],data[:,1],label = label)
                label = textfile[i+2].strip()
                N = 0

        if not N:
            N = i + 1
            continue
        textfile[i] = element_list

    data = np.array(textfile[N:i])
    ax.plot(data[:,0],data[:,1],label = label)
    ax.legend()
    return fig,ax

def del_AfterMath_files():
    """Delete the redundant files created when exporting data in AfterMath."""
    file_list = os.listdir()
    for file in file_list:
        if file[-3:] != 'csv':
            continue
        file_parts = file.split('_')
        if file_parts[-1] != 'Current vs Potential.csv':
            os.remove(file)
            continue
        os.rename(file,file_parts[-4])

def time_from_potential(potential, scanrate):

    '''potential will rise and fall, but time must increase linearly and monotonically.
    To convert potential array to time array we
    1 Mirror the decreasing potential segments about the x axis so they are increasing.
    2 Shift each potential segment to form a continuous line'''
    size = potential.size
    time = np.zeros(size)
    time[0] = potential[0]
    rising = True #whether potential is increasing or decreasing
    delta = 0 #differenece between time line and the increasing potential segment
    for i in range(1,size):
        if potential[i] < potential[i-1]: #falling --> time = - potential (mirrors potential about the x axis)
            if rising: # if potential was previously rising, delta changes from mirror operation
                delta = time[i-1] + potential[i-1]
            time[i] = -potential[i] + delta
            rising = False
        else: #rising --> time = potential
            if not rising:
                delta = time[i-1] - potential[i-1]
            time[i] = potential[i] + delta
            rising = True
    time -= time[0]
    time /= scanrate
    return time

### THIS FUNCTION HAS BEEN UPDATED BELOW
# def potential_from_time_and_CV_limits(time_arr,upper_lim,lower_lim,scan_rt,init_dir,
#                                       start_E):
#     """Create an Array of potential points from an Array of time points and parameters for a CV experiment."""
#     potential = np.zeros(time_arr.size)
#     potential[0] = start_E
#     direction = init_dir
#     if start_E > upper_lim:
#         first_cyc_index_below_lim = (((start_E - upper_lim) / scan_rt)
#                                            / (time_arr[1] - time_arr[0]))
#     elif start_E < lower_lim:
#         first_cyc_index_below_lim = (((lower_lim - start_E) / scan_rt)
#                                            / (time_arr[1] - time_arr[0]))
#     else:
#         first_cyc_index_below_lim = 0
#     for i in range(1,time_arr.size):
#         dt = time_arr[i] - time_arr[i-1]
#         if direction == 'down':
#             v =  potential[i-1] - dt * scan_rt
#         if direction == 'up':
#             v =  potential[i-1] + dt * scan_rt
#         if v < lower_lim and i > first_cyc_index_below_lim:
#             direction = 'up'
#         if v > upper_lim and i > first_cyc_index_below_lim:
#             direction = 'down'

#         potential[i] = v
#     return potential

def potential_from_time_and_CV_limits(time_arr,upper_lim,lower_lim,scan_rt,init_dir,start_E):
    '''
    Compute the potential waveform as a fucntion of time.

    Parameters
    ----------
    time_arr : ndarray, shape (N)
        An array of time points.
    upper_lim : float
        upper potential limit, Eh.
    lower_lim : float
        lower potential limit, El.
    scan_rt : float
        scan rate.
    init_dir : str
        "up" or "down" for anodic or cathodic scan directions, respectively.
    start_E : float
        Initail potential.

    Returns
    -------
    E : ndarray, shape (N)
        The potential waveform.

    '''
    E = np.zeros(time_arr.size)
    t_half_cycle = (upper_lim - lower_lim) / scan_rt

    for i in range(time_arr.size):
        t = time_arr[i]

        # CASE 1: Ei = Eh
        if start_E == upper_lim:
            if (t//t_half_cycle)%2 == 0: #even number of half cycles
                E[i] = upper_lim - (t % t_half_cycle) * scan_rt
            if (t//t_half_cycle)%2 == 1: #odd number of half cycles
                E[i] = lower_lim + (t % t_half_cycle) * scan_rt

        # CASE 2: Ei = El
        elif start_E == lower_lim:
            if (t//t_half_cycle)%2 == 1: #even number of half cycles
                E[i] = upper_lim - (t % t_half_cycle) * scan_rt
            if (t//t_half_cycle)%2 == 0: #odd number of half cycles
                E[i] = lower_lim + (t % t_half_cycle) * scan_rt

        # CASE 3: Ei > Eh
        elif start_E > upper_lim:
            t_to_Eh = (start_E - upper_lim) / scan_rt
            if t < t_to_Eh:
                E[i] = start_E - t* scan_rt
            else: # CASE 1
                t -= t_to_Eh
                if (t//t_half_cycle)%2 == 0: #even number of half cycles
                    E[i] = upper_lim - (t % t_half_cycle) * scan_rt
                if (t//t_half_cycle)%2 == 1: #odd number of half cycles
                    E[i] = lower_lim + (t % t_half_cycle) * scan_rt

        # CASE 4: Ei < El
        elif start_E < lower_lim:
            t_to_El = (lower_lim - start_E) / scan_rt
            if t < t_to_El:
                E[i] = start_E + t* scan_rt
            else: # CASE 2
                t -= t_to_El
                if (t//t_half_cycle)%2 == 1: #even number of half cycles
                    E[i] = upper_lim - (t % t_half_cycle) * scan_rt
                if (t//t_half_cycle)%2 == 0: #odd number of half cycles
                    E[i] = lower_lim + (t % t_half_cycle) * scan_rt

        # CASE 5: Ei < Eh and initail scan diraction is down
        elif (start_E < upper_lim) and (init_dir == 'down'):
            t_to_El = (start_E - lower_lim) / scan_rt
            if t < t_to_El:
                E[i] = start_E - t* scan_rt
            else: # CASE 2
                t -= t_to_El
                if (t//t_half_cycle)%2 == 1: #even number of half cycles
                    E[i] = upper_lim - (t % t_half_cycle) * scan_rt
                if (t//t_half_cycle)%2 == 0: #odd number of half cycles
                    E[i] = lower_lim + (t % t_half_cycle) * scan_rt

        # CASE 6: Ei < Eh and initail scan diraction is down
        elif (start_E > lower_lim) and (init_dir == 'up'):
            t_to_Eh = (upper_lim - start_E) / scan_rt
            if t < t_to_Eh:
                E[i] = start_E + t* scan_rt
            else: # CASE 1
                t -= t_to_Eh
                if (t//t_half_cycle)%2 == 0: #even number of half cycles
                    E[i] = upper_lim - (t % t_half_cycle) * scan_rt
                if (t//t_half_cycle)%2 == 1: #odd number of half cycles
                    E[i] = lower_lim + (t % t_half_cycle) * scan_rt
    return E

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def linearize_region(arr,i_start,i_end):
    '''
    Replace data points between indicies i_start and i_end with points from
    a line containing i_start and i_end.

    Parameters
    ----------
    arr : array
        array who's data will be modified
    i_start : TYPE
        index of first point of line
    i_end : TYPE
        index of second point of line

    Returns
    -------
    None.

    '''
    m = (arr[i_end] - arr[i_start]) / (i_end - i_start)
    arr[i_start +1:i_end] = np.linspace(1,
                                        i_end - i_start - 1,
                                        i_end - i_start - 1) * m + arr[i_start]

def vertical_line_at_x(x,ax,**kwarg):
    lim = ax.get_ylim()
    ax.plot([x,x],lim,'r--',**kwarg)
    ax.set_ylim(lim)
