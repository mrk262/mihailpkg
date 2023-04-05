# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:20:33 2022

@author: Mihail
"""

import numpy as np

def get_cycling_data(filename):
    '''
    Load relavent galvanostatic cycling data from excel sheet into an array and save the array.

    Parameters
    ----------
    filename : str
        File name of raw tab delimited ASCII data.

    Returns
    -------
    cycling_data : ndarray
        Data cols are time, voltage, cycleID, and current.

    '''


    delim = '\t'
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


def get_CE(data_arr, tol=1e-4):
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
    tol = tol
    def get_switching_times(data_arr, tol=1e-5):
        '''Find the time points where the cycle number changes '''
        switching_times = []
        prev_cyc_curr = 0. # tracks changes in current, allows 0 current OCP
        for i in range(data_arr.shape[0]): #loop over the rows in the array
            curr = data_arr[i,3]
            if abs(curr - prev_cyc_curr) > tol: # if the cycle ID changes, the end of the cycle has been reached
                if abs(data_arr[i+1,3] - curr) > tol: continue # probable outlier point
                time = data_arr[i,0]
                switching_times.append(time)
                prev_cyc_curr = curr
        return np.array(switching_times)

    switching_times = get_switching_times(data_arr)
    n = switching_times.size - 1 # first element is 0 only for initialization
    CE = np.zeros(n//2) # number of full cycles
    for i in range(1,n,2): #only need every other half cycle
        charging_time = switching_times[i] - switching_times[i-1]
        discharge_time = switching_times[i+1] - switching_times[i]
        CE[i//2] = discharge_time/charging_time
    return CE
