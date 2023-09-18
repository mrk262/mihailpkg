import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tkinter as tk
from .functions import average, savitzky_golay, antideriv, parse_file, time_from_potential

class CV:
    """
    Create a CV from a data array.

    default kwargs: t_col=0, t_units='s',
                    E_col=1, E_units='V vs Li/Li⁺',
                    i_col=2, i_units='A',
                    Q_units='mC/cm²', delimiter='\\t'

    """

    def __init__(self,
                 data_array,
                 scanrate = 0,
                 electrode_area = 0,
                 t_col = 0,
                 E_col = 1,
                 i_col = 2,
                 label = '',
                 **kwargs):

        defaultKwargs = {'t_units':'s', 'E_units':'V vs Li/Li$^+$', 'i_units':'A', 'Q_units':'mC/cm$^2$',
                         'delimiter':'\t'}
        kwargs = { **defaultKwargs, **kwargs }


        def set_current(i):
            if electrode_area:
                self.current = data_array[:,i_col-i] / electrode_area
                self.current_units = kwargs['i_units'] + '/cm$^2$'
            else:
                self.current = data_array[:,i_col-i]
                self.current_units = kwargs['i_units']

        #Load the data into an array
        if type(data_array) == str:
            data_array = parse_file(data_array, delim=kwargs['delimiter'])
        elif type(data_array) == np.ndarray:
            data_array = np.copy(data_array)
        else:
            print('Cannot convert data array into CV')

        if t_col == None: #3 or more column data but none are time
            new_array = np.zeros((data_array.shape[0],2))
            new_array[:,0] =  data_array[:,E_col]
            new_array[:,1] =  data_array[:,i_col]
            data_array = new_array
            E_col = 1
            i_col = 2

        size,N_COLS = data_array.shape

        if N_COLS == 2: # No time data provided
            self.potential = data_array[:,E_col-1]
            self.potential_units = kwargs['E_units']
            set_current(1)
            if scanrate: # calculate time from scanrate and potential
                self.time = time_from_potential(self.potential, scanrate)
                self.time_units = kwargs['t_units']
                self.calculate_charge()
            else: # not enough information to calculate time and charge
                self.time = None
                self.time_units = 'None'
                self.charge = None
                self.charge_units = 'None'

        else: # time data provided
            self.time = data_array[:,t_col]
            self.time_units = kwargs['t_units']
            self.potential = data_array[:,E_col]
            self.potential_units = kwargs['E_units']
            set_current(0)
            self.calculate_charge()

        self.label = label
        self.scanrate = scanrate
        self.area = electrode_area

    def __repr__(self):
        # time and charge have default units even if the arrays are not defined
        string1 = 'CV object: {}\n'.format(self.label)
        s1,s2 = '[{}]'.format(self.potential_units).replace('$^+$', '⁺') , '[{}]'.format(self.current_units).replace('$\mu$', 'µ').replace('$^2$', '²')
        string2 = '\tPot:  {:<15s} Cur:    {:<15s}\n'.format(s1,s2)
        s1,s2 = '[{}]'.format(self.time_units) , '[{}]'.format(self.charge_units).replace('$\mu$', 'µ').replace('$^2$', '²')
        string3 = '\tTime: {:<15s} Charge: {:<15s}'.format(s1,s2)

        return string1 + string2 + string3

    def gui(self):

        def update_instance():
            self.label = labelEntry.get()
            root.title(self.label + ' : ' + str(id(self)))

        root = tk.Tk()
        root.title(self.label + ' : ' + str(id(self)))
        #root.geometry('200x100')
        frame = tk.Frame(root)
        frame.pack()

        tk.Button(master=frame, text='Update', padx=20, command=update_instance).grid(row=0,column=1,columnspan = 2)

        labelLabel = tk.Label(master=frame, text='label: ', padx=20)
        labelLabel.grid(row=1,column=1,columnspan = 1)
        labelEntry = tk.Entry(master=frame)
        labelEntry.insert(0,self.label)
        labelEntry.grid(row=1,column=2,columnspan = 1)



    @property
    def size(self):
        if self.potential.size == self.current.size:
            return self.potential.size
        else:
            return print('Size mismatch')

    @property
    def time_label(self):
        return 'Time / ' + self.time_units

    @property
    def potential_label(self):
        return 'Potential / ' + self.potential_units

    @property
    def xlabel(self):
        return 'Potential / ' + self.potential_units

    @property
    def current_label(self):
        return 'Current / ' + self.current_units

    @property
    def ylabel(self):
        return 'Current / ' + self.current_units

    @property
    def charge_label(self):
        return 'Charge / ' + self.charge_units


    def calculate_charge(self):
        try:
            self.charge = antideriv(self.time,self.current)
            if len(self.current_units.split('/')) == 1:
                self.charge_units = (self.current_units
                                     + '.'
                                     + self.time_units)
            if len(self.current_units.split('/')) == 2:
                self.charge_units = (self.current_units.split('/')[0].strip()
                                     + '.'
                                     + self.time_units
                                     + '/'
                                     + self.current_units.split('/')[1].strip())
        except TypeError:
            print('Need time or scan rate to calculate charge.')

    def smooth(self, window_size=21, order=3, deriv=0):
        try: self.time = savitzky_golay(self.time, window_size, order, deriv)
        except TypeError: pass
        self.potential = savitzky_golay(self.potential, window_size, order, deriv)
        self.current = savitzky_golay(self.current, window_size, order, deriv)
        try: self.charge = savitzky_golay(self.charge, window_size, order, deriv)
        except TypeError: pass

    def average_points(self,n):
        try: self.time = average(self.time, n)
        except TypeError: pass
        self.potential = average(self.potential, n)
        self.current = average(self.current, n)
        try: self.charge = average(self.charge, n)
        except TypeError: pass

    def average_cycles(self, visualize = False): #challanging because each cycle can be a different number of pts
        if visualize: fig,ax = plt.subplots()
        cycles = []
        if visualize: indicies = self.cycles(visualize = True, gradient=True, tol=0.02)[0]
        else: indicies = self.cycles(plot=False, tol=0.02)
        current = np.zeros(indicies[1])
        for cycle_number in range(1,len(indicies)):
            i_s = indicies[cycle_number -1]
            i_f = indicies[cycle_number]
            cycle = self.current[i_s:i_f]
            cycles.append(cycle)
            try:
                current += cycle
                if visualize: ax.plot(cycle, label = 'cycle {}'.format(cycle_number))
            except ValueError:
                print('Adjusting cycle {}'.format(cycle_number))
                if current.size > cycle.size:
                    previous_cycle = cycles[cycle_number - 2]
                    adjusted_cycle = np.zeros(current.size)
                    adjusted_cycle[:cycle.size] = cycle
                    adjusted_cycle[cycle.size:current.size] = previous_cycle[cycle.size:current.size]
                    current += adjusted_cycle
                    if visualize: ax.plot(adjusted_cycle, label = 'cycle {}'.format(cycle_number))
                elif current.size < cycle.size and ((cycle.size - current.size) / current.size) < 0.001:
                    #less than 0.1% error in size descrepency
                    current += cycle[:current.size]
                else:
                    print('Error, cycle size discrepancy')
                    print(current.size,cycle.size)
                    current += cycle[:current.size]
        current /= len(cycles)
        if visualize:
            ax.plot(current,label = 'Average')
            ax.legend()
        self.current = current
        self.potential = self.potential[:current.size]

    def peak_integration(self, lower_index, upper_index, **plot_kwargs):
        '''
        Calculate the charge under a peak defined by indicies with linear background

        Parameters
        ----------
        lower_index : int
            Index of beggining of peak.
        upper_index : int
            Index of end of peak.
        **plot_kwargs : keywords
            Keyword arguments for plot function

        Returns
        -------
        peak_area : float
            Charge between linear background and current peak.
        fig : Figure
            Plot of integrated peak and background for integration.

        '''

        fig,ax = plt.subplots() #plot current
        potential, current, charge = self.potential[lower_index:upper_index], self.current[lower_index:upper_index], self.charge[lower_index:upper_index] - self.charge[lower_index]
        background = 0.5 * (current[0] + current[-1]) * (potential[-1] - potential[0]) / self.scanrate
        peak_area = charge[-1] - background
        ax.plot(potential, current, **plot_kwargs)
        ax.plot([potential[0],potential[-1]],[current[0],current[-1]], 'r:', linewidth=1)
        ax.set_xlabel(self.potential_label)
        ax.set_ylabel(self.current_label)
        ax.legend()
        fig.text(0.1,1.0,'Area = {:.0f} $\mu$C/cm$^2$'.format(peak_area))
        return peak_area, fig


    def plot(self,ax = False,i_s = 0,i_f = 'end',cycle = 0, warn = True,
             label = '',**kwargs):
        """Plot the CV on the given axes or otherwise create and return fig,ax."""
        def label_mismatch():
            if len(ax.lines) > 0 and warn:
                if ax.get_xlabel() != 'Potential / ' + self.potential_units:
                    print('Mismatching Potential Units???')
                if ax.get_ylabel() != 'Current / ' + self.current_units:
                    print('Mismatching Current Units???')

        if i_f == 'end':
            i_f = self.size
        if cycle:
            indicies = self.cycles(plot=False)
            i_s = indicies[cycle -1]
            i_f = indicies[cycle]
        if ax:
            label_mismatch()
            ax.plot(self.potential[i_s:i_f],
                    self.current[i_s:i_f],
                    label = label,**kwargs)
            if label:
                ax.legend()

        else:
            fig,ax = plt.subplots()
            ax.plot(self.potential[i_s:i_f],
                    self.current[i_s:i_f],
                    label = label,
                    **kwargs)
            ax.set_xlabel('Potential / ' + self.potential_units)
            ax.set_ylabel('Current / ' + self.current_units)
            if label:
                ax.legend()
            return fig,ax

    def cycles(self, tol = 0.005, plot = True, ax=False, visualize = False, last = True, gradient = False):
        """
        Determine indicies of each cycle in CV.

        Take a Cyclic Voltammogram (CV) having multiple cycles and find the row
        indicies corresponding to the end of each cycle. Optionally, plot the CV
        with the first cycle in blue and subsequent cycles going from red to black.

        Parameters
        ----------
        tol : number
            Noise tolerance. Passed as prominence=tol to scipy's
            find_peaks function.
        plot : Boolean
            True if a plot of the data with with cycle number visualized by color
            gradation is desired.
        last : Boolean
            True if the final cycle is incomplete and desired to be plotted.

        Returns
        -------
        cycle_indicies : numpy array
            Array containing the indicies of the rows in CV where each cycle ends.
        """
        def append_endpts():
            nonlocal cycle_indicies
            yo = [0]
            for i in cycle_indicies:
                yo.append(int(i))
            yo.append(self.size-1)
            cycle_indicies = yo

        arr = -1 * np.abs(self.potential - self.potential[0])
        cycle_indicies,_ = find_peaks(arr, prominence=tol)
        arr1 = self.potential
        if (arr1[np.argmax(arr1)] - np.abs(arr1[0]) > tol
            ) or -1*(arr1[np.argmax(arr1)] - np.abs(arr1[0]) > tol):

            cycle_indicies = cycle_indicies[1::2]

        if plot:
            return_none = False
            if ax:
                ax = ax
                return_none = True
            else:
                fig,ax = plt.subplots()

            if visualize:
                fig0,ax0 = plt.subplots()
                ax0.plot(self.potential)
                ax0.plot(cycle_indicies,np.zeros(cycle_indicies.size)
                         + self.potential[0],'ro')

            if cycle_indicies.size == 0:
                ax.plot(self.potential,
                        self.current)
            else:

                for i in range(cycle_indicies.size - 1):
                    cycle_start = cycle_indicies[i]
                    cycle_end = cycle_indicies[i+1]
                    if gradient:
                        ax.plot(self.potential[cycle_start:cycle_end],
                                self.current[cycle_start:cycle_end],
                                color = (1 - i/cycle_indicies.size,0,0))
                    else:
                        ax.plot(self.potential[cycle_start:cycle_end],
                                self.current[cycle_start:cycle_end],
                                color = (0.6,0.6,1,1))

                if last:
                    cycle_start = cycle_indicies[-1]
                    if gradient:

                        ax.plot(self.potential[cycle_start:],
                                self.current[cycle_start:],
                                color = (0,0,0))
                    else:
                        ax.plot(self.potential[cycle_start:],
                                self.current[cycle_start:],
                                color = (0.6,0.6,1,1))

                first_scan_end = cycle_indicies[0]
                ax.plot(self.potential[:first_scan_end],
                        self.current[:first_scan_end],
                        color = (0,0,1))
                ax.set_xlabel(self.potential_label)
                ax.set_ylabel(self.current_label)

            if return_none:
                append_endpts()
                return cycle_indicies
            append_endpts()
            return cycle_indicies,fig,ax
        append_endpts()
        return cycle_indicies


    def clip_data(self,i_s=0,i_f='end'):
        """Remove data points, who's index is not between i_s and i_f, from the CV."""
        if i_f == 'end':
            i_f = self.size
        try:
            self.time = self.time[i_s:i_f]
        except: pass
        try:
            self.charge = self.charge[i_s:i_f]
        except: pass
        self.potential = self.potential[i_s:i_f]
        self.current = self.current[i_s:i_f]

    def scale_time(self, factor):
        try:
            self.time *= factor
        except TypeError:
            print('Time data does not exist')

        #Update time_units
        qty = self.time_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.time_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.time_units = prefix + qty[1]
            except: print('error')

    def scale_potential(self, factor):
        self.potential *= factor

        #Update potential_units
        qty = self.potential_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.potential_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.potential_units = prefix + qty[1]
            except: print('error')

    def scale_current(self, factor):
        self.current *= factor

        #Update current_units
        qty = self.current_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.current_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.current_units = prefix + qty[1]
            except: print('error')

    def scale_charge(self, factor):
        try:
            self.charge *= factor
        except TypeError:
            print('Charge data does not exist')

        #Update charge_units
        qty = self.charge_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.charge_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.charge_units = prefix + qty[1]
            except: print('error')

    def shift_time(self, shift):
        try:
            self.time += shift
        except TypeError:
            print('Time data does not exist')

    def shift_potential(self, shift):
        self.potential += shift

    def shift_current(self, shift):
        self.current += shift

    def shift_charge(self, shift):
        try:
            self.charge += shift
        except TypeError:
            print('Charge data does not exist')

    def set_time_units(self,units):
        '''
        1 --> min
        2 --> hr
        '''
        if type(units) == str:
            self.time_units = units
        else:
            if units == 1:
                self.time_units = 'min'
            elif units == 2:
                self.time_units = 'hr'
            else:
                print('Error!')

    def set_potential_units(self,units):
        '''
        1 --> V vs Li/Li$^+$
        2 --> V vs Ag/AgCl
        '''
        if type(units) == str:
            self.potential_units = units
        else:
            if units == 1:
                self.potential_units = 'V vs Li/Li$^+$'
            elif units == 2:
                self.potential_units = 'V vs Ag/AgCl'
            else:
                print('Error!')

    def set_current_units(self,units):
        '''
        1 --> mA/cm$^2$
        2 --> $\mu$A/cm$^2$
        '''
        if type(units) == str:
            self.current_units = units
        else:
            if units == 1:
                self.current_units = 'mA/cm$^2$'
            elif units == 2:
                self.current_units = '$\mu$A/cm$^2$'
            else:
                print('Error!')

    def set_charge_units(self,units):
        '''
        1 --> mC/cm$^2$
        2 --> $\mu$C/cm$^2$
        3 --> C/cm$^2$
        '''
        if type(units) == str:
            self.charge_units = units
        else:
            if units == 1:
                self.charge_units = 'mC/cm$^2$'
            elif units == 2:
                self.charge_units = '$\mu$C/cm$^2$'
            elif units == 3:
                self.charge_units = 'C/cm$^2$'
            else:
                print('Error!')
