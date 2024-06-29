# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:20:33 2022

@author: Mihail
"""

import numpy as np

def get_cycling_data(filename, delim='\t'):
    '''
    Load relavent galvanostatic cycling data from excel sheet into an array and save the array.

    Parameters
    ----------
    filename : str
        File name of 'Records' sheet from battery cycler.

    Returns
    -------
    cycling_data : ndarray
        Data cols: | time | voltage | cycleID | current |

    '''


    delim = delim
    time_col = 5 # hr:min:s
    voltage_col = 7
    cycleID_col = 1
    current_col = 6
    cycling_data = []
    with open(filename, 'r') as file:
        for i,line in enumerate(file):
            if i == 0: #column labels and units are in the first row
                col_title_list = line.split(delim)
                header = (col_title_list[time_col] + delim +
                          col_title_list[voltage_col] + delim +
                          col_title_list[cycleID_col] + delim +
                          col_title_list[current_col] + delim)
            if i > 0: #data is in subsequent rows
                time_h_m_s = line.split(delim)[time_col].split(':')
                time_s = (float(time_h_m_s[0])*60**2 +
                          float(time_h_m_s[1])*60 +
                          float(time_h_m_s[2]))
                voltage = float(line.split(delim)[voltage_col])
                cycle_n = float(line.split(delim)[cycleID_col])
                current = float(line.split(delim)[current_col])
                cycling_data.append([time_s, voltage, cycle_n, current])
    cycling_data = np.array(cycling_data)
    np.savetxt(filename[:-4] + 'Processed.txt', cycling_data,
               header=header, delimiter=delim)
    return cycling_data


def get_CE(data_arr, tol=1e-5, capacity=False):
    '''
    Calculate the current effeciency based on ratio of discharge to charging time

    Parameters
    ----------
    data_arr : ndarray
        Cycling data, with cycle number in the 3rd col.

    tol : float
        Tolerance for determining change in cycle direction / A

    Returns
    -------
    ndarray
        Current effeciency of each cycle.

    '''
    def get_switching_times(data_arr, tol=tol):
        '''Find the time points where the cycle number changes '''
        switching_times = []
        prev_cyc_curr = 0. # tracks changes in current, allows 0 current OCP
        for i in range(data_arr.shape[0]): #loop over the rows in the array
            curr = data_arr[i,3]
            if abs(curr - prev_cyc_curr) > tol: # if the cycle ID changes, the end of the cycle has been reached
                if i < data_arr.shape[0]-1: # fix bug where cycle ID changes in the final row
                    if abs(data_arr[i+1,3] - curr) > tol: continue # probable outlier point
                time = data_arr[i,0]
                switching_times.append(time)
                prev_cyc_curr = curr
        switching_times.append(data_arr[-1,0]) # current does not switch at end of final cucle
        return np.array(switching_times)

    switching_times = get_switching_times(data_arr)
    n = switching_times.size - 1 # first element is 0 only for initialization
    if capacity: CE = np.zeros(((n//2),3))
    else: CE = np.zeros(n//2) # number of full cycles
    for i in range(1,n,2): #only need every other half cycle
        charging_time = switching_times[i] - switching_times[i-1]
        discharge_time = switching_times[i+1] - switching_times[i]
        if capacity:
            CE[i//2,0] = discharge_time/charging_time
            CE[i//2,1] = discharge_time
            CE[i//2,2] = charging_time
        else:
            CE[i//2] = discharge_time/charging_time
    return CE

def get_dchg_profile(data_arr, cycle=1, tol=1e-5, capacity=False):
    '''
    Calculate the current effeciency based on ratio of discharge to charging time

    Parameters
    ----------
    data_arr : ndarray
        Cycling data, with cycle number in the 3rd col.

    tol : float
        Tolerance for determining change in cycle direction / A

    Returns
    -------
    ndarray
        Current effeciency of each cycle.

    '''
    data_arr = data_arr.copy()
    def get_switching_indicies(data_arr, tol=tol):
        '''Find the time points where the cycle number changes '''
        switching_indicies = []
        prev_cyc_curr = 0. # tracks changes in current, allows 0 current OCP
        for i in range(data_arr.shape[0]): #loop over the rows in the array
            curr = data_arr[i,3]
            if abs(curr - prev_cyc_curr) > tol: # if the cycle ID changes, the end of the cycle has been reached
                if i < data_arr.shape[0]-1: # fix bug where cycle ID changes in the final row
                    if abs(data_arr[i+1,3] - curr) > tol: continue # probable outlier point
                switching_indicies.append(i)
                prev_cyc_curr = curr
        switching_indicies.append(data_arr.shape[0]-1)
        return np.array(switching_indicies)

    switching_indicies = get_switching_indicies(data_arr)
    charge = slice(switching_indicies[2*cycle-1], switching_indicies[2*cycle]-1)
    discharge = slice(switching_indicies[2*cycle-2], switching_indicies[2*cycle-1]-1)

    charge_profile = data_arr[charge,:2]
    discharge_profile = data_arr[discharge,:2]

    discharge_profile[:,0], charge_profile[:,0] = discharge_profile[:,0] - discharge_profile[0,0], charge_profile[:,0] - discharge_profile[0,0]
    charge_profile[:,0] = -1*charge_profile[:,0] + 2*charge_profile[0,0]

    return discharge_profile, charge_profile
