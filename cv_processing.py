# -*- coding: utf-8 -*-
"""
Created on Thu May 13 00:19:48 2021

@author: Mihail
"""
from .functions import *
from .CV import CV
from .EQCM import EQCM
from .axis_labels import axis_labels

def CV_from_numbers(file_dictionary,n,t_col = False,E_col=0,i_col=1):
    """Create a CV object from an entry in a file dictionary."""

    def CV_from_n(file_dictionary,n,t_col = False,E_col=0,i_col=1):
        keys = [*file_dictionary]
        if t_col:
            time = file_dictionary[keys[n]][:,t_col]
            potential = file_dictionary[keys[n]][:,E_col]
            current = file_dictionary[keys[n]][:,i_col]
            cv = CV(np.stack((time,potential,current), axis = 1),
                    label = keys[n])
            return cv
        else:
            potential = file_dictionary[keys[n]][:,E_col]
            current = file_dictionary[keys[n]][:,i_col]
            cv = CV(np.stack((potential,current),axis = 1),
                    label = keys[n])
            return cv

    if type(n) == int:
        cv = CV_from_n(file_dictionary, n,
                        t_col = t_col,
                        E_col = E_col,
                        i_col = i_col)
        return cv

    if type(n) == str:
        cv_list = []
        for i in range(len(file_dictionary)):
            cv = CV_from_n(file_dictionary, i,
                            t_col = t_col,
                            E_col = E_col,
                            i_col = i_col)
            cv_list.append(cv)
        return cv_list

    else:
        cv_list = []
        for i in n:
            cv = CV_from_n(file_dictionary, i,
                            t_col = t_col,
                            E_col = E_col,
                            i_col = i_col)
            cv_list.append(cv)
        return cv_list


# Clean up namespace
del np
del os
del pickle
del plt
#del pd
del fft
del dp
del fd
del tk


# depreciated 2023 01 30

# import numpy as np
# import pandas as pd
# #import matplotlib
# import matplotlib.pyplot as plt
# from scipy.fft import rfft, irfft, fftfreq
# from scipy.signal import find_peaks
# import os
# import pickle
# import mihailpkg.data_plotter as dp

# def average(arr, n):
#     group_size = n
#     number_of_groups = arr.size//group_size
#     averaged_arr = np.zeros(number_of_groups)
#     for i in range(number_of_groups):
#         averaged_arr[i] = np.average(arr[i*group_size:(i+1)*group_size])
#     return averaged_arr

# def run(file, path):
#     '''
#     Execute a python script in its directory

#     Parameters
#     ----------
#     file : str
#         Name of python script.
#     path : str
#         Directory containing the script.

#     Returns
#     -------
#     None.

#     '''
#     def run_local(file, path):
#         os.chdir(path)
#         exec(open(file).read())
#     CWD = os.getcwd()
#     run_local(file, path)
#     os.chdir(CWD)

# def mass_to_charge_cont(mass, charge, npts_to_avg = 50):
#     '''


#     Parameters
#     ----------
#     mass : ndarray
#         Array of mass values from experiment.
#     charge : ndarray
#         Array of charge values from experiment.
#     npts_to_avg : int, optional
#         The number of points used in the moving difference. The default is 50.

#     Returns
#     -------
#     m2c_cont : ndarray
#         Differential mass to charge ratio based on a moving difference over the
#         mass and charge arrays.
#     slice_name : slice
#         Since a moving difference requires a truncated array, use this slice
#         object to slice data for plotting the mass to charge ratio

#     '''
#     charge = resize_array(charge, mass.size) #resize charge array to be compatable with mass array

#     diff_mass_cont = np.zeros(mass.size - npts_to_avg) #differential mass using moving difference
#     diff_charge_cont = np.zeros(mass.size - npts_to_avg) #differential charge using moving difference
#     for i in range(mass.size - npts_to_avg):
#         diff_mass_cont[i] = mass[i + npts_to_avg] - mass[i]
#         diff_charge_cont[i] = charge[i + npts_to_avg] - charge[i]

#     m2c_cont = -diff_mass_cont * 96485 / diff_charge_cont
#     slice_name = slice(round(npts_to_avg/2),-round(npts_to_avg/2))
#     return m2c_cont, slice_name


# def PAC(L, LAMBDA, RG):
#     '''
#     Model ET kinetics dependant PAC feedback current.

#     For details see doi: https://doi.org/10.1002/cphc.200900600

#     Parameters
#     ----------
#     L : ndarray
#         Tip-substrate distance normalized by electrode radius.
#     LAMBDA : float
#         ET rate constant normalised by D/a.
#     RG : float
#         Tip radius normalized by electrode radius.

#     Returns
#     -------
#     ndarray
#         Tip current normalized by current at infinite distance.

#     '''
#     def Ni_cond(L, RG):
#         def alpha(RG):
#             return (np.log(2)
#                     + np.log(2) * (1 - 2/np.pi * np.arccos(1/RG))
#                     - np.log(2) * (1 - (2/np.pi * np.arccos(1/RG))**2))
#         def beta(RG):
#             return (1
#                     + 0.639 * (1 - 2/np.pi * np.arccos(1/RG))
#                     - 0.186 * (1 - (2/np.pi * np.arccos(1/RG))**2))
#         return (alpha(RG)
#                 + np.pi / (4*beta(RG)*np.arctan(L))
#                 + (1 - alpha(RG) - 1/(2*beta(RG))) * (2/np.pi*np.arctan(L)))

#     def Ni_ins(L, RG):
#         return ((2.08/RG**0.358 * (L - 0.145/RG) + 1.585)
#                 / (2.08/RG**0.358 * (L + 0.0023*RG) + 1.57 + np.log(RG)/L + 2/(np.pi*RG) * np.log(1 + np.pi*RG/(2*L))))

#     return (Ni_cond((L + 1/LAMBDA), RG)
#             + (Ni_ins(L,RG) - 1)
#             / ((1 + 2.47*RG**0.31*L*LAMBDA)*(1 + L**(0.006*RG + 0.113)*LAMBDA**(-0.0236*RG + 0.91))))

# def linear_interpolation(y1, y2, x1, x2, x):
#     m = (y2 - y1) / (x2 - x1)
#     y = m * (x - x1) + y1
#     return y

# def fft_low_pass(array,cutoff_freq,baseline = 0, visualize = False):
#     yf = rfft(array) # discrete Fourier Transform
#     xf = fftfreq(array.size) # sample frequencies
#     cut_f_signal = yf.copy()
#     cut_f_signal[index_of_value(cutoff_freq,xf):] = baseline * array.size / 2 # remove frequencies above cutoff_freq
#     cut_signal = irfft(cut_f_signal)
#     if visualize:
#         fig,ax = plt.subplots(ncols=2, figsize=(8,4))
#         ax[0].plot(array, label='Original signal')
#         ax[0].plot(cut_signal, label = 'Processed signal')
#         ax[0].set_xlabel('array index / i')
#         ax[0].legend()
#         ax[1].plot(2/array.size*np.abs(yf), label='FFT Original')
#         ax[1].plot(2/array.size*np.abs(cut_f_signal), label='FFT Processes')
#         ax[1].set_xlabel('sampling frequency / i$^{-1}$')
#         ax[0].legend()
#         return cut_signal
#     return cut_signal

# def index_of_value(value,array):
#     return np.argmin(np.abs(array - value))

# def dict_value_from_num(dict,n):
#     return dict[list(dict.keys())[n]]

# def resize_array(arr,newsize):
#     arr1 = np.copy(arr)
#     arr2 = np.zeros(newsize)
#     i_prev = 0
#     j_prev = 0
#     for i in range(1, arr1.size):
#         j = i * arr2.size / arr1.size
#         m2 = (arr1[i] - arr1[i_prev]) / (j - j_prev)
#         for k in range((round(j) - round(j_prev))):
#             j_int = round(j_prev + k)
#             arr2[j_int] = arr1[i] + (j_int - j) * m2
#         i_prev = i
#         j_prev = j
#     for n in range(j_int,arr2.size): #extrapolate any pts leftover
#         arr2[n] = arr1[i] + (n - j) * m2
#     return arr2

# class axis_labels:

#     Ag_E_label = 'Potential / V vs Ag/AgCl'
#     Li_E_label = 'Potential / V vs Li/Li$^+$'
#     i_umA_label = 'Current / $\mu$A/cm$^2$'
#     i_mA_label = 'Current / mA/cm$^2$'
#     Z_real_label = 'Z$_{real}$ / $\Omega$'
#     Z_imag_label = 'Z$_{imag}$ / $\Omega$'


# def text_figure(desc, fig=None, height=0.2, fontsize=10):
#     if fig and height < 1:
#         axes = fig.get_axes()
#         for ax in axes:
#             ax.axis('off')
#     if not fig:
#         fig,ax = plt.subplots(figsize = (8,6))
#         ax.axis('off')
#     fig.text(0,height,desc,fontsize = fontsize, family = 'consolas')
#     fig.tight_layout()
#     return fig

# def generate_figures(n):
#     fig_refs = []
#     ax_refs = []
#     for i in range(n):
#         fig,ax = plt.subplots()
#         fig_refs.append(fig)
#         ax_refs.append(ax)
#     return fig_refs,ax_refs

# def load_and_plot_files():
#     """Use the data plotter to load data files into a file dictionary."""
#     dp.main()
#     print('-'*20)
#     for i,key in enumerate(dp.data):
#         print(i,'\t',key)
#     print('-'*20)
#     data = dp.data.copy()
#     dp.data = {'label':{}}
#     dp.label = dp.data['label']
#     return data

# def convert_all_txt_files(delimiter = '\t'):
#     data = {'label':{}}
#     label = data['label']
#     def parse_file(filename):
#         """Parse the data file for numeric data and append data array to the dictionary."""

#         with open(filename) as file:
#             text_file = file.readlines()
#         numeric_data = []
#         delim = delimiter

#         for i in range(1,100):
#             try:
#                 last_row = text_file[-1*i].strip().split(delim)
#                 for element in last_row:
#                     float(element)
#                 num_col = len(last_row)
#                 if num_col < 2:
#                     continue
#                 break
#             except:
#                 continue

#         for i,row in enumerate(text_file):
#             columns_list = row.strip().split(delim)
#             for j,element in enumerate(columns_list):
#                 try:                                            #if the row contains the correct number of elements and these are
#                     numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
#                     columns_list[j] = numeric_element
#                     if j == num_col - 1:
#                         numeric_data.append(columns_list)
#                 except:
#                     if len(columns_list) == num_col and j == 0: #if the (text) row contains the same number of elements as a data row,
#                         label[filename] = columns_list          #it must be a label describing the data
#                     continue
#         data_array = np.array(numeric_data)
#         data[filename[:-4]] = data_array                         #add the data to the dictionary
#         if filename not in label:
#             label[filename] = list(range(num_col))

#     for filename in os.listdir():
#         if filename[-3:] == 'txt':
#             try:
#                 parse_file(filename)
#             except:
#                 continue
#     with open('preprocessed data.pickle','wb') as file:
#         pickle.dump(data, file)

# def SECM_1Dto2D(nx3_array):

#     data_array = nx3_array
#     x = data_array[:,0]
#     y = data_array[:,1]
#     z = data_array[:,2]
#     xperiod = np.argmax(x)
#     xmax = x[xperiod]
#     yperiod = np.argmax(y)
#     ymax = y[yperiod]
#     if xperiod < yperiod: #longaxis = 'y'
#         xstep = x[1] - x[0]
#         ystep = y[xperiod + 1] - y[0]

#     if xperiod > yperiod: #longaxis = 'x'
#         ystep = y[1] - y[0]
#         xstep = x[yperiod + 1] - x[0]

#     xpts = int(xmax/xstep + 1)
#     xaxis = np.linspace(x[0],xmax,xpts)
#     ypts = int(ymax/ystep + 1)
#     yaxis = np.linspace(y[0],ymax,ypts)
#     X,Y = np.meshgrid(xaxis,yaxis)
#     Z = np.empty(xpts * ypts)
#     Z[:] = np.nan
#     Z[:z.size] = z
#     Z = Z.reshape(xpts,ypts)
#     return X,Y,Z

# def array_from_number(data,n,alias = False):
#     keys = [*data]
#     if alias:
#         array = data[keys[n]]
#     if not alias:
#         array = np.copy(data[keys[n]])
#     return array


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
#                         'filename1':[data_col1,data_col2,...],
#                         'filename2':[data_col1,data_col2,...],
#                         ...}

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
#                                   columns=label_dictionary[key],
#                                   copy=True)
#         dataFrame.filename = key
#         return dataFrame

#     if type(identifier) == list:
#         dataFrame_list = []
#         for idnum in identifier:
#             key = filename_list[idnum]
#             dataFrame = pd.DataFrame(data= file_dictionary[key],
#                                       columns=label_dictionary[key],
#                                       copy=True)
#             dataFrame.filename = key
#             dataFrame_list.append(dataFrame)
#         return dataFrame_list

#     if type(identifier) == str:
#         if identifier == 'all':
#             dataFrame_list = []
#             for idnum in range(1,len(filename_list)):
#                 key = filename_list[idnum]
#                 dataFrame = pd.DataFrame(data= file_dictionary[key],
#                                           columns=label_dictionary[key],
#                                           copy=True)
#                 dataFrame.filename = key
#                 dataFrame_list.append(dataFrame)
#             return dataFrame_list

#         else:
#             dataFrame_list = []
#             for filename in filename_list:
#                 if identifier in filename:
#                     dataFrame = pd.DataFrame(data= file_dictionary[filename],
#                                               columns=label_dictionary[filename],
#                                               copy=True)
#                     dataFrame.filename = key
#                     dataFrame_list.append(dataFrame)
#             return dataFrame_list

# class EQCM:
#     """
#     Create a CV from a data array.
#     """
#     def __init__(self,
#                   data_array,
#                   qcm_area = 0.4,
#                   C_f = 42,
#                   t_col = 5,
#                   t_units = 's',
#                   f_col = 2,
#                   f_units = 'Hz',
#                   R_col = 1,
#                   R_units = '$\Omega$',
#                   label = '',
#                   delimiter = '\t'):
#         def parse_file(filename):
#             """Parse the data file for numeric data and append data array to the dictionary."""

#             with open(filename) as file:
#                 text_file = file.readlines()
#             numeric_data = []
#             delim = delimiter

#             for i in range(1,100):
#                 try:
#                     last_row = text_file[-1*i].strip().split(delim)
#                     for element in last_row:
#                         float(element)
#                     num_col = len(last_row)
#                     if num_col < 2:
#                         continue
#                     break
#                 except:
#                     continue

#             for i,row in enumerate(text_file):
#                 columns_list = row.strip().split(delim)
#                 for j,element in enumerate(columns_list):
#                     try:                                            #if the row contains the correct number of elements and these are
#                         numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
#                         columns_list[j] = numeric_element
#                         if j == num_col - 1:
#                             numeric_data.append(columns_list)
#                     except: continue
#             data_array = np.array(numeric_data)
#             return data_array

#         if type(data_array) == str:
#             data_array = parse_file(data_array)
#         elif type(data_array) == np.ndarray:
#             data_array = np.copy(data_array)
#         else:
#             print('Cannot convert data array into CV')

#         size,N_COLS = data_array.shape
#         self.size = size
#         self.time = data_array[:,t_col]
#         self.time_units = t_units
#         self.time_label = 'Time / ' + self.time_units
#         self.freq = data_array[:,f_col]
#         self.freq_units = f_units
#         self.freq_label = '$\Delta$freq / ' + self.freq_units
#         self.res = data_array[:,R_col]
#         self.res_units = R_units
#         self.res_label = 'Resistance / ' + self.res_units
#         self.mass = -1*self.freq / C_f
#         if f_units == 'Hz':
#             self.mass_label = 'Mass / $\mu$g/cm$^2$'
#         else:
#             print('Check mass units')
#         self.area = qcm_area
#         self.label = label

#     def scale_time(self, factor):
#         self.time *= factor

#     def scale_freq(self, factor):
#         self.freq *= factor

#     def scale_res(self, factor):
#         self.res *= factor

#     def scale_mass(self, factor):
#         self.mass *= factor

#     def shift_time(self, shift):
#         self.time += shift

#     def shift_freq(self, shift):
#         self.freq += shift

#     def shift_res(self, shift):
#         self.res += shift

#     def shift_mass(self, shift):
#         self.mass += shift

#     def clip_data(self,i_s=0,i_f='end'):
#         """Remove data points, who's index is not between i_s and i_f, from the CV."""
#         if i_f == 'end':
#             i_f = self.size
#         self.time = self.time[i_s:i_f]
#         self.freq = self.freq[i_s:i_f]
#         self.res = self.res[i_s:i_f]
#         self.mass = self.mass[i_s:i_f]
#         self.size = i_f - i_s

#     def plot(self,i_s=0,i_f='end'):
#         FIG,AX = plt.subplots(nrows=2, ncols=1, sharex=True,figsize = (5,8),tight_layout=False)
#         FIG.subplots_adjust(hspace=0)
#         if i_f == 'end':
#             i_f = self.size
#         AX[0].plot(self.time[i_s:i_f],self.freq[i_s:i_f])
#         AX[1].plot(self.time[i_s:i_f],self.res[i_s:i_f])
#         AX[0].set_ylabel(self.freq_label)
#         AX[1].set_ylabel(self.res_label)
#         AX[1].set_xlabel(self.time_label)
#         return FIG,AX

#     def mass_to_charge(self,time,charge):
#         '''expect charge in uC/cm2
#         return mass to charge ratio in g/mol'''
#         if time.size > self.size:
#             m_to_c = np.zeros(self.size)
#             for i,t in enumerate(self.time):
#                 index = np.argmin(np.abs(time - t))
#                 m_to_c[i] = ((self.mass[i] * 96485)
#                                       / (charge[index] *-1))
#             return self.time, m_to_c

#         if self.size > time.size:
#             m_to_c = np.zeros(time.size)
#             for i,t in enumerate(time):
#                 index = np.argmin(np.abs(self.time - t))
#                 m_to_c[i] = ((self.mass[index] * 96485)
#                                       / (charge[i] *-1))
#             return time, m_to_c

#     def set_freq_units(self,units):
#         '''
#         1 --> kHz
#         '''
#         if type(units) == str:
#             self.freq_units = units
#         else:
#             if units == 1:
#                 self.freq_units = 'kHz'
#             else:
#                 print('Error!')
#         self.freq_label = '$\Delta$freq / ' + self.freq_units

#     def set_time_units(self,units):
#         '''
#         1 --> min
#         2 --> hr
#         '''
#         if type(units) == str:
#             self.time_units = units
#         else:
#             if units == 1:
#                 self.time_units = 'min'
#             elif units == 2:
#                 self.time_units = 'hr'
#             else:
#                 print('Error!')
#         self.time_label ='Time / ' + self.time_units


# class CV:
#     """
#     Create a CV from a data array.
#     """

#     def __init__(self,
#                   data_array,
#                   scanrate = 0,
#                   electrode_area = 0,
#                   t_col = 0,
#                   t_units = 's',
#                   E_col = 1,
#                   E_units = 'V vs Li/Li$^+$',
#                   i_col = 2,
#                   i_units = 'A',
#                   label = '',
#                   delimiter = '\t'):

#         def parse_file(filename):
#             """Parse the data file for numeric data and append data array to the dictionary."""

#             with open(filename) as file:
#                 text_file = file.readlines()
#             numeric_data = []
#             delim = delimiter

#             for i in range(1,100):
#                 try:
#                     last_row = text_file[-1*i].strip().split(delim)
#                     for element in last_row:
#                         float(element)
#                     num_col = len(last_row)
#                     if num_col < 2:
#                         continue
#                     break
#                 except:
#                     continue

#             for i,row in enumerate(text_file):
#                 columns_list = row.strip().split(delim)
#                 for j,element in enumerate(columns_list):
#                     try:                                            #if the row contains the correct number of elements and these are
#                         numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
#                         columns_list[j] = numeric_element
#                         if j == num_col - 1:
#                             numeric_data.append(columns_list)
#                     except: continue
#             data_array = np.array(numeric_data)
#             return data_array

#         def set_current(i):
#             if electrode_area:
#                 self.current = data_array[:,i_col-i] / electrode_area
#                 self.current_units = i_units + '/cm$^2$'
#             else:
#                 self.current = data_array[:,i_col-i]
#                 self.current_units = i_units

#         if type(data_array) == str:
#             data_array = parse_file(data_array)
#         elif type(data_array) == np.ndarray:
#             data_array = np.copy(data_array)
#         else:
#             print('Cannot convert data array into CV')

#         if t_col == None: #3 or more col data but none are time
#             new_array = np.zeros((data_array.shape[0],2))
#             new_array[:,0] =  data_array[:,E_col]
#             new_array[:,1] =  data_array[:,i_col]
#             data_array = new_array
#             E_col = 1
#             i_col = 2

#         size,N_COLS = data_array.shape

#         if N_COLS == 2:
#             self.potential = data_array[:,E_col-1]
#             self.potential_units = E_units
#             set_current(1)
#             if scanrate:
#                 '''potential will rise and fall, but time must increase linearly and monotonically.
#                 To convert potential array to time array we
#                 1 Mirror the decreasing potential segments about the x axis so they are increasing.
#                 2 Shift each potential segment to form a continuous line'''
#                 self.time = np.zeros(size)
#                 self.time[0] = self.potential[0]
#                 rising = True #whether potential is increasing or decreasing
#                 delta = 0 #differenece between time line and the increasing potential segment
#                 for i in range(1,size):
#                     if self.potential[i] < self.potential[i-1]: #falling --> time = - potential (mirrors potential about the x axis)
#                         if rising: # if potential was previously rising, delta changes from mirror operation
#                             delta = self.time[i-1] + self.potential[i-1]
#                         self.time[i] = -self.potential[i] + delta
#                         rising = False
#                     else: #rising --> time = potential
#                         if not rising:
#                             delta = self.time[i-1] - self.potential[i-1]
#                         self.time[i] = self.potential[i] + delta
#                         rising = True
#                 self.time -= self.time[0]
#                 self.time /= scanrate
#                 self.time_units = t_units
#                 self.time_label = 'Time / ' + self.time_units
#                 self.calculate_charge()

#         else:
#             self.time = data_array[:,t_col]
#             self.time_units = t_units
#             self.time_label = 'Time / ' + self.time_units


#             self.potential = data_array[:,E_col]
#             self.potential_units = E_units

#             set_current(0)

#             self.calculate_charge()


#         self.label = label
#         self.scanrate = scanrate
#         self.size = size
#         if electrode_area:
#             self.area = electrode_area
#         self.xlabel = 'Potential / ' + self.potential_units
#         self.ylabel = 'Current / ' + self.current_units
#         self.potential_label = self.xlabel
#         self.current_label = self.ylabel


#     def calculate_charge(self):
#         try:
#             self.charge = antideriv(self.time,self.current)
#             if len(self.current_units.split('/')) == 1:
#                 self.charge_units = (self.current_units
#                                       + '.'
#                                       + self.time_units)
#             if len(self.current_units.split('/')) == 2:
#                 self.charge_units = (self.current_units.split('/')[0].strip()
#                                       + '.'
#                                       + self.time_units
#                                       + '/'
#                                       + self.current_units.split('/')[1].strip())
#             self.charge_label = 'Charge / ' + self.charge_units
#         except NameError:
#             print('Need time points for current measurement or scan rate')

#     def scale_time(self, factor):
#         self.time *= factor

#     def scale_potential(self, factor):
#         self.potential *= factor

#     def scale_current(self, factor):
#         self.current *= factor

#     def scale_charge(self, factor):
#         try:
#             self.charge *= factor
#         except NameError:
#             print('Charge data does not exist')

#     def shift_time(self, shift):
#         self.time += shift

#     def shift_potential(self, shift):
#         self.potential += shift

#     def shift_current(self, shift):
#         self.current += shift

#     def set_time_units(self,units):
#         '''
#         1 --> min
#         2 --> hr
#         '''
#         if type(units) == str:
#             self.time_units = units
#         else:
#             if units == 1:
#                 self.time_units = 'min'
#             elif units == 2:
#                 self.time_units = 'hr'
#             else:
#                 print('Error!')
#         self.time_label ='Time / ' + self.time_units

#     def set_potential_units(self,units):
#         '''
#         1 --> V vs Li/Li$^+$
#         2 --> V vs Ag/AgCl
#         '''
#         if type(units) == str:
#             self.potential_units = units
#         else:
#             if units == 1:
#                 self.potential_units = 'V vs Li/Li$^+$'
#             elif units == 2:
#                 self.potential_units = 'V vs Ag/AgCl'
#             else:
#                 print('Error!')
#         self.xlabel ='Potential / ' + self.potential_units
#         self.potential_label = self.xlabel

#     def set_current_units(self,units):
#         '''
#         1 --> mA/cm$^2$
#         2 --> $\mu$A/cm$^2$
#         '''
#         if type(units) == str:
#             self.current_units = units
#         else:
#             if units == 1:
#                 self.current_units = 'mA/cm$^2$'
#             elif units == 2:
#                 self.current_units = '$\mu$A/cm$^2$'
#             else:
#                 print('Error!')
#         self.ylabel ='Current / ' + self.current_units
#         self.current_label = self.ylabel

#     def set_charge_units(self,units):
#         '''
#         1 --> mC/cm$^2$
#         2 --> $\mu$C/cm$^2$
#         3 --> C/cm$^2$
#         '''
#         if type(units) == str:
#             self.charge_units = units
#         else:
#             if units == 1:
#                 self.charge_units = 'mC/cm$^2$'
#             elif units == 2:
#                 self.charge_units = '$\mu$C/cm$^2$'
#             elif units == 3:
#                 self.charge_units = 'C/cm$^2$'
#             else:
#                 print('Error!')
#         self.charge_label ='Charge / ' + self.charge_units

#     def smooth(self, window_size=21, order=3, deriv=0):
#         try: self.time = savitzky_golay(self.time, window_size, order, deriv)
#         except AttributeError: pass
#         try: self.potential = savitzky_golay(self.potential, window_size, order, deriv)
#         except AttributeError: pass
#         try: self.current = savitzky_golay(self.current, window_size, order, deriv)
#         except AttributeError: pass

#     def average_points(self,n):
#         try: self.time = average(self.time, n)
#         except AttributeError: pass
#         try: self.potential = average(self.potential, n)
#         except AttributeError: pass
#         try: self.current = average(self.current, n)
#         except AttributeError: pass


#     def average_cycles(self, visualize = False): #challanging because each cycle can be a different number of pts
#         if visualize: fig,ax = plt.subplots()
#         cycles = []
#         if visualize: indicies = self.cycles(visualize = True, gradient=True, tol=0.02)[0]
#         else: indicies = self.cycles(plot=False, tol=0.02)
#         current = np.zeros(indicies[1])
#         for cycle_number in range(1,len(indicies)):
#             i_s = indicies[cycle_number -1]
#             i_f = indicies[cycle_number]
#             cycle = self.current[i_s:i_f]
#             cycles.append(cycle)
#             try:
#                 current += cycle
#                 if visualize: ax.plot(cycle, label = 'cycle {}'.format(cycle_number))
#             except ValueError:
#                 print('Adjusting cycle {}'.format(cycle_number))
#                 if current.size > cycle.size:
#                     previous_cycle = cycles[cycle_number - 2]
#                     adjusted_cycle = np.zeros(current.size)
#                     adjusted_cycle[:cycle.size] = cycle
#                     adjusted_cycle[cycle.size:current.size] = previous_cycle[cycle.size:current.size]
#                     current += adjusted_cycle
#                     if visualize: ax.plot(adjusted_cycle, label = 'cycle {}'.format(cycle_number))
#                 elif current.size < cycle.size and ((cycle.size - current.size) / current.size) < 0.001:
#                     #less than 0.1% error in size descrepency
#                     current += cycle[:current.size]
#                 else:
#                     print('Error, cycle size discrepancy')
#                     print(current.size,cycle.size)
#                     current += cycle[:current.size]
#         current /= len(cycles)
#         if visualize:
#             ax.plot(current,label = 'Average')
#             ax.legend()
#         self.current = current
#         self.potential = self.potential[:current.size]

#     def peak_integration(self, lower_index, upper_index, **plot_kwargs):
#         '''
#         Calculate the charge under a peak defined by indicies with linear background

#         Parameters
#         ----------
#         lower_index : int
#             Index of beggining of peak.
#         upper_index : int
#             Index of end of peak.
#         **plot_kwargs : keywords
#             Keyword arguments for plot function

#         Returns
#         -------
#         peak_area : float
#             Charge between linear background and current peak.
#         fig : Figure
#             Plot of integrated peak and background for integration.

#         '''

#         fig,ax = plt.subplots() #plot current
#         potential, current, charge = self.potential[lower_index:upper_index], self.current[lower_index:upper_index], self.charge[lower_index:upper_index] - self.charge[lower_index]
#         background = 0.5 * (current[0] + current[-1]) * (potential[-1] - potential[0]) / self.scanrate
#         peak_area = charge[-1] - background
#         ax.plot(potential, current, **plot_kwargs)
#         ax.plot([potential[0],potential[-1]],[current[0],current[-1]], 'r:', linewidth=1)
#         ax.set_xlabel(self.potential_label)
#         ax.set_ylabel(self.current_label)
#         ax.legend()
#         fig.text(0.1,1.0,'Area = {:.0f} $\mu$C/cm$^2$'.format(peak_area))
#         return peak_area, fig


#     def plot(self,ax = False,i_s = 0,i_f = 'end',cycle = 0, warn = True,
#               label = '',**kwargs):
#         """Plot the CV on the given axes or otherwise create and return fig,ax."""
#         def label_mismatch():
#             if len(ax.lines) > 0 and warn:
#                 if ax.get_xlabel() != 'Potential / ' + self.potential_units:
#                     print('Mismatching Potential Units???')
#                 if ax.get_ylabel() != 'Current / ' + self.current_units:
#                     print('Mismatching Current Units???')

#         if i_f == 'end':
#             i_f = self.size
#         if cycle:
#             indicies = self.cycles(plot=False)
#             i_s = indicies[cycle -1]
#             i_f = indicies[cycle]
#         if ax:
#             label_mismatch()
#             ax.plot(self.potential[i_s:i_f],
#                     self.current[i_s:i_f],
#                     label = label,**kwargs)
#             if label:
#                 ax.legend()

#         else:
#             fig,ax = plt.subplots()
#             ax.plot(self.potential[i_s:i_f],
#                     self.current[i_s:i_f],
#                     label = label,
#                     **kwargs)
#             ax.set_xlabel('Potential / ' + self.potential_units)
#             ax.set_ylabel('Current / ' + self.current_units)
#             if label:
#                 ax.legend()
#             return fig,ax

#     def cycles(self, tol = 0.005, plot = True, ax=False, visualize = False, last = True, gradient = False):
#         """
#         Determine indicies of each cycle in CV.

#         Take a Cyclic Voltammogram (CV) having multiple cycles and find the row
#         indicies corresponding to the end of each cycle. Optionally, plot the CV
#         with the first cycle in blue and subsequent cycles going from red to black.

#         Parameters
#         ----------
#         tol : number
#             Noise tolerance. Passed as prominence=tol to scipy's
#             find_peaks function.
#         plot : Boolean
#             True if a plot of the data with with cycle number visualized by color
#             gradation is desired.
#         last : Boolean
#             True if the final cycle is incomplete and desired to be plotted.

#         Returns
#         -------
#         cycle_indicies : numpy array
#             Array containing the indicies of the rows in CV where each cycle ends.
#         """
#         def append_endpts():
#             nonlocal cycle_indicies
#             yo = [0]
#             for i in cycle_indicies:
#                 yo.append(int(i))
#             yo.append(self.size-1)
#             cycle_indicies = yo

#         arr = -1 * np.abs(self.potential - self.potential[0])
#         cycle_indicies,_ = find_peaks(arr, prominence=tol)
#         arr1 = self.potential
#         if (arr1[np.argmax(arr1)] - np.abs(arr1[0]) > tol
#             ) or -1*(arr1[np.argmax(arr1)] - np.abs(arr1[0]) > tol):

#             cycle_indicies = cycle_indicies[1::2]

#         if plot:
#             return_none = False
#             if ax:
#                 ax = ax
#                 return_none = True
#             else:
#                 fig,ax = plt.subplots()

#             if visualize:
#                 fig0,ax0 = plt.subplots()
#                 ax0.plot(self.potential)
#                 ax0.plot(cycle_indicies,np.zeros(cycle_indicies.size)
#                           + self.potential[0],'ro')

#             if cycle_indicies.size == 0:
#                 ax.plot(self.potential,
#                         self.current)
#             else:

#                 for i in range(cycle_indicies.size - 1):
#                     cycle_start = cycle_indicies[i]
#                     cycle_end = cycle_indicies[i+1]
#                     if gradient:
#                         ax.plot(self.potential[cycle_start:cycle_end],
#                                 self.current[cycle_start:cycle_end],
#                                 color = (1 - i/cycle_indicies.size,0,0))
#                     else:
#                         ax.plot(self.potential[cycle_start:cycle_end],
#                                 self.current[cycle_start:cycle_end],
#                                 color = (0.6,0.6,1,1))

#                 if last:
#                     cycle_start = cycle_indicies[-1]
#                     if gradient:

#                         ax.plot(self.potential[cycle_start:],
#                                 self.current[cycle_start:],
#                                 color = (0,0,0))
#                     else:
#                         ax.plot(self.potential[cycle_start:],
#                                 self.current[cycle_start:],
#                                 color = (0.6,0.6,1,1))

#                 first_scan_end = cycle_indicies[0]
#                 ax.plot(self.potential[:first_scan_end],
#                         self.current[:first_scan_end],
#                         color = (0,0,1))
#                 ax.set_xlabel(self.potential_label)
#                 ax.set_ylabel(self.current_label)

#             if return_none:
#                 append_endpts()
#                 return cycle_indicies
#             append_endpts()
#             return cycle_indicies,fig,ax
#         append_endpts()
#         return cycle_indicies


#     def clip_data(self,i_s=0,i_f='end'):
#         """Remove data points, who's index is not between i_s and i_f, from the CV."""
#         if i_f == 'end':
#             i_f = self.size
#         if hasattr(self, 'time'):
#             self.time = self.time[i_s:i_f]
#         if hasattr(self, 'charge'):
#             self.charge = self.charge[i_s:i_f]
#         self.potential = self.potential[i_s:i_f]
#         self.current = self.current[i_s:i_f]
#         self.size = i_f - i_s


# def CV_from_numbers(file_dictionary,n,t_col = False,E_col=0,i_col=1):
#     """Create a CV object from an entry in a file dictionary."""

#     def CV_from_n(file_dictionary,n,t_col = False,E_col=0,i_col=1):
#         keys = [*file_dictionary]
#         if t_col:
#             time = file_dictionary[keys[n]][:,t_col]
#             potential = file_dictionary[keys[n]][:,E_col]
#             current = file_dictionary[keys[n]][:,i_col]
#             cv = CV(np.stack((time,potential,current), axis = 1),
#                     label = keys[n])
#             return cv
#         else:
#             potential = file_dictionary[keys[n]][:,E_col]
#             current = file_dictionary[keys[n]][:,i_col]
#             cv = CV(np.stack((potential,current),axis = 1),
#                     label = keys[n])
#             return cv

#     if type(n) == int:
#         cv = CV_from_n(file_dictionary, n,
#                         t_col = t_col,
#                         E_col = E_col,
#                         i_col = i_col)
#         return cv

#     if type(n) == str:
#         cv_list = []
#         for i in range(len(file_dictionary)):
#             cv = CV_from_n(file_dictionary, i,
#                             t_col = t_col,
#                             E_col = E_col,
#                             i_col = i_col)
#             cv_list.append(cv)
#         return cv_list

#     else:
#         cv_list = []
#         for i in n:
#             cv = CV_from_n(file_dictionary, i,
#                             t_col = t_col,
#                             E_col = E_col,
#                             i_col = i_col)
#             cv_list.append(cv)
#         return cv_list

# def antideriv(x,y):
#     """Take 2 Arrays x and y = f(x) and return Array z = F(x) = int(x0,x,y)."""
#     N = y.size
#     z = np.zeros(N)
#     for i in range(1,N):
#         z[i] = z[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
#     return z

# def differentiate(y ,x=None, npts=1):
#     """Differentiate y array using n point moving difference"""
#     dydx = np.zeros(y.size - npts)
#     if type(x) == type(None):
#         for i in range(dydx.size):
#             dydx[i] = y[i + npts] - y[i]
#         return dydx
#     elif x.size == y.size:
#         for i in range(dydx.size):
#             dy = y[i + npts] - y[i]
#             dx = x[i + npts] - x[i]
#             dydx[i] = dy/dx
#         return dydx, x[npts//2:dydx.size + npts//2]
#     else:
#         print('Error, x and y arrays must be same size')

# def copy_labels(fig_or_ax,source_fig_or_ax):
#     """Copy axes labels from one figure to another."""
#     if 'Axes' in str(type(fig_or_ax)):
#         fig_or_ax.set_xlabel(source_fig_or_ax.get_xlabel())
#         fig_or_ax.set_ylabel(source_fig_or_ax.get_ylabel())

#     if 'Figure' in str(type(fig_or_ax)):
#         ax0 = fig_or_ax.get_axes()[0]
#         ax1 = source_fig_or_ax.get_axes()[0]

#         ax0.set_xlabel(ax1.get_xlabel())
#         ax0.set_ylabel(ax1.get_ylabel())

# def fig_to_data(fig,filename):
#     """
#     Create a .mlg data file from a figure.

#     Takes a figure with multiple lines plotted on a single axes and generates
#     a multi line graph (.mlg) file. The file structure is

#     X units    Y units
#     line0 label
#     x0    y0
#     x1    y1
#     x2    y2
#     ...
#     line1 label
#     x0    y0
#     x1    y1
#     x2    y2


#     Parameters
#     ----------
#     fig : Figure
#         Figure with one axes containing data lines
#     filename : string
#         name of file created, no extension

#     Returns
#     -------
#     None.

#     """
#     with open(filename + '.mlg','w') as file:
#         ax = fig.get_axes()[0]
#         x_label = ax.get_xlabel()
#         y_label = ax.get_ylabel()
#         file.write(x_label + '\t' + y_label + '\n')
#         for line in ax.lines:
#             label = line.get_label()
#             file.write(label + '\n')
#             xdata,ydata = line.get_data()
#             for i in range(xdata.size):
#                 file.write(str(xdata[i]) + '\t' + str(ydata[i]) + '\n')

#     fig.savefig(filename + '.png')

# def data_to_fig(filename):
#     """Open a multiple line graph file as a figure."""
#     with open(filename + '.mlg','r') as file:
#         textfile = file.readlines()

#     fig,ax = plt.subplots()
#     ax.set_xlabel(textfile[0].split('\t')[0].strip())
#     ax.set_ylabel(textfile[0].split('\t')[1].strip())
#     label = textfile[1].strip()

#     N=2 #tracks index of start of line data sequence and used to
#         #incriment outer (row) loop when end of data sequence is reached
#     for i,row in enumerate(textfile[2:]):
#         element_list = row.split('\t')

#         for j,element in enumerate(element_list):

#             try:
#                 numeric_element = float(element)
#                 element_list[j] = numeric_element

#             except ValueError:

#                 data = np.array(textfile[N:i])
#                 ax.plot(data[:,0],data[:,1],label = label)
#                 label = textfile[i+2].strip()
#                 N = 0

#         if not N:
#             N = i + 1
#             continue
#         textfile[i] = element_list

#     data = np.array(textfile[N:i])
#     ax.plot(data[:,0],data[:,1],label = label)
#     ax.legend()
#     return fig,ax

# def del_AfterMath_files():
#     """Delete the redundant files created when exporting data in AfterMath."""
#     file_list = os.listdir()
#     for file in file_list:
#         if file[-3:] != 'csv':
#             continue
#         file_parts = file.split('_')
#         if file_parts[-1] != 'Current vs Potential.csv':
#             os.remove(file)
#             continue
#         os.rename(file,file_parts[1])


# def potential_from_time_and_CV_limits(time_arr,upper_lim,lower_lim,scan_rt,init_dir,
#                                       start_E):
#     """Create an Array of potential points from an Array of time points and parameters for a CV experiment."""
#     potential = np.zeros(time_arr.size)
#     potential[0] = start_E
#     direction = init_dir
#     if start_E > upper_lim:
#         first_cyc_index_below_lim = (((start_E - upper_lim) / scan_rt)
#                                             / (time_arr[1] - time_arr[0]))
#     elif start_E < lower_lim:
#         first_cyc_index_below_lim = (((lower_lim - start_E) / scan_rt)
#                                             / (time_arr[1] - time_arr[0]))
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

# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#     The Savitzky-Golay filter removes high frequency noise from data.
#     It has the advantage of preserving the original shape and
#     features of the signal better than other types of filtering
#     approaches, such as moving averages techniques.
#     Parameters
#     ----------
#     y : array_like, shape (N,)
#         the values of the time history of the signal.
#     window_size : int
#         the length of the window. Must be an odd integer number.
#     order : int
#         the order of the polynomial used in the filtering.
#         Must be less then `window_size` - 1.
#     deriv: int
#         the order of the derivative to compute (default = 0 means only smoothing)
#     Returns
#     -------
#     ys : ndarray, shape (N)
#         the smoothed signal (or it's n-th derivative).
#     Notes
#     -----
#     The Savitzky-Golay is a type of low-pass filter, particularly
#     suited for smoothing noisy data. The main idea behind this
#     approach is to make for each point a least-square fit with a
#     polynomial of high order over a odd-sized window centered at
#     the point.
#     Examples
#     --------
#     t = np.linspace(-4, 4, 500)
#     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#     ysg = savitzky_golay(y, window_size=31, order=4)
#     import matplotlib.pyplot as plt
#     plt.plot(t, y, label='Noisy signal')
#     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#     plt.plot(t, ysg, 'r', label='Filtered signal')
#     plt.legend()
#     plt.show()
#     References
#     ----------
#     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#         Data by Simplified Least Squares Procedures. Analytical
#         Chemistry, 1964, 36 (8), pp 1627-1639.
#     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#         W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#         Cambridge University Press ISBN-13: 9780521880688
#     """
#     import numpy as np
#     from math import factorial

#     try:
#         window_size = np.abs(np.int(window_size))
#         order = np.abs(np.int(order))
#     except ValueError:
#         raise ValueError("window_size and order have to be of type int")
#     if window_size % 2 != 1 or window_size < 1:
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order+1)
#     half_window = (window_size -1) // 2
#     # precompute coefficients
#     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#     # pad the signal at the extremes with
#     # values taken from the signal itself
#     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#     y = np.concatenate((firstvals, y, lastvals))
#     return np.convolve( m[::-1], y, mode='valid')

# def linearize_region(arr,i_start,i_end):
#     '''
#     Replace data points between indicies i_start and i_end with points from
#     a line containing i_start and i_end.

#     Parameters
#     ----------
#     arr : array
#         array who's data will be modified
#     i_start : TYPE
#         index of first point of line
#     i_end : TYPE
#         index of second point of line

#     Returns
#     -------
#     None.

#     '''
#     m = (arr[i_end] - arr[i_start]) / (i_end - i_start)
#     arr[i_start +1:i_end] = np.linspace(1,
#                                         i_end - i_start - 1,
#                                         i_end - i_start - 1) * m + arr[i_start]

# def vertical_line_at_x(x,ax):
#     lim = ax.get_ylim()
#     ax.plot([x,x],lim,'r--',linewidth = 1)
#     ax.set_ylim(lim)
