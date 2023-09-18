
import numpy as np
import matplotlib.pyplot as plt
from .functions import average, savitzky_golay, parse_file, resize_array, text_figure

plt.plot

class EQCM:
    '''
    Create EQCM object.

    Parameters
    ----------
    data_array : str OR ndarray
        Data filename or array.
    analog : bool, optional
        Sets default column positions. The default is False.
    t_col : int, optional
        Time [s] column number in data array. The default is 5.
    f_col : int, optional
       Frequency [Hz] column number in data array. The default is 2.
    R_col : int, optional
        Resistance [Ω] column number in data array. The default is 1.
    E_col : int, optional
        Potential [V vs Li/Li⁺] column number in data array. The default is None.
    i_col : int, optional
        Current [µA/cm²] column number in data array. The default is None.
    Q_col : int, optional
        Charge [mC/cm²] column number in data array. The default is None.
    label : str, optional
        Experiment description. The default is ''.
    **kwargs : str or float
        Set *_units* for the variables.
        Set *delimiter*. The default is \t.
        Set *C_f*. The default is 42 Hz.cm²/µg.
        Set *qcm_area*. The default is 0.4 cm².



    Returns
    -------
    None.

    '''
    def __init__(self,
                 data_array,
                 analog = False,
                 t_col = 5,
                 f_col = 2,
                 R_col = 1,
                 E_col = None,
                 i_col = None,
                 Q_col = None,
                 label = '',
                 **kwargs):

        defaultKwargs = {'t_units':'s', 'f_units':'Hz', 'R_units':'$\Omega$',
                         'E_units':'V vs Li/Li$^+$', 'i_units':'$\mu$A/cm$^2$', 'Q_units':'mC/cm$^2$',
                         'C_f':42, 'delimiter':'\t', 'qcm_area':0.4,}
        kwargs = { **defaultKwargs, **kwargs }

        if analog:
            t_col, E_col, i_col, f_col, R_col = 0, 1, 2, 3, 4
            with open(data_array, 'r') as file:
                labels = file.readline()
                labels = file.readline().split('\t')
                units = [label.split(' / ')[1] for label in labels]
                kwargs['t_units'], kwargs['E_units'], kwargs['i_units'], kwargs['f_units'], kwargs['R_units'] = units[0], units[1], units[2], units[3], units[4]

        if type(data_array) == str:
            data_array = parse_file(data_array, delim=kwargs['delimiter'])
        elif type(data_array) == np.ndarray:
            data_array = np.copy(data_array)
        else:
            print('Cannot convert data array into EQCM object')

        size,N_COLS = data_array.shape
        self.time = data_array[:,t_col]
        self.time_units = kwargs['t_units']
        self.freq = data_array[:,f_col]
        self.freq_units = kwargs['f_units']
        self.res = data_array[:,R_col]
        self.res_units = kwargs['R_units']
        self.potential = None if E_col == None else data_array[:,E_col]
        self.potential_units = 'None' if E_col == None else kwargs['E_units']
        self.current = None if i_col == None else data_array[:,i_col]
        self.current_units = 'None' if E_col == None else kwargs['i_units']
        self.charge = None if Q_col == None else data_array[:,Q_col]
        self.charge_units = 'None' if E_col == None else kwargs['Q_units']
        self.area = kwargs['qcm_area']
        self.C_f = kwargs['C_f']
        self.label = label

    def __repr__(self):
        string1 = 'EQCM object: {}\n'.format(self.label)
        s1,s2,s3 = '[{}]'.format(self.time_units), '[{}]'.format(self.freq_units), '[{}]'.format(self.res_units).replace('$\Omega$', 'Ω')
        string2 = '\tTime: {:<15s} Freq: {:<15s} Res: {:<15s}\n'.format(s1,s2,s3)
        s1,s2,s3 = '[{}]'.format(self.potential_units).replace('$^+$', '⁺'), '[{}]'.format(self.current_units).replace('$\mu$', 'µ').replace('$^2$', '²'), '[{}]'.format(self.charge_units).replace('$\mu$', 'µ').replace('$^2$', '²')
        string3 = '\tPot:  {:<15s} Cur: {:<15s}  Charge: {:<15s}'.format(s1,s2,s3)
        return string1 + string2 + string3

    @property
    def time_step(self):
        return (self.time[-1] - self.time[0]) / (self.size - 1)

    @property
    def mass(self):
        return -1*self.freq / self.C_f

    @property
    def size(self):
        if self.time.size == self.freq.size == self.res.size:
            return self.freq.size
        else:
            return print('Size mismatch')

    @property
    def time_label(self):
        return 'Time / ' + self.time_units

    @property
    def freq_label(self):
        return '$\Delta$freq / ' + self.freq_units

    @property
    def mass_label(self):
        return 'Mass / $\mu$g/cm$^2$' if self.freq_units == 'Hz' else print('Check mass units')

    @property
    def res_label(self):
        return 'Resistance / ' + self.res_units

    @property
    def potential_label(self):
        return 'Potential / ' + self.potential_units

    @property
    def current_label(self):
        return 'Current / ' + self.current_units

    @property
    def charge_label(self):
        return 'Charge / ' + self.charge_units

    def load_cv_data(self,cv,i_s=0, i_f='end', verify=False):
        '''
        Clip excess EQCM data

        Parameters
        ----------
        cv : TYPE
            DESCRIPTION.
        i_s : TYPE, optional
            DESCRIPTION. The default is 0.
        i_f : TYPE, optional
            DESCRIPTION. The default is 'end'.

        Returns
        -------
        None.

        '''

        if verify:
            fig,ax = plt.subplots(nrows=2, figsize=(4,6), sharex=True)
            text_figure(self.label + '\n' + cv.label, fig=fig, height=1)
            ax[0].set_ylabel(self.freq_label)
            ax[1].set_ylabel(cv.potential_label)
            ax[1].set_xlabel(self.time_label)
            ax[0].plot(self.time, self.freq, label='raw')
            ax[1].plot(cv.time, cv.potential, label='raw')

        if i_f == 'end': i_f = self.size
        self.clip_data(i_s=i_s, i_f=i_f)   # trim data
        self.shift_time(-self.time[0])
        self.shift_freq(-self.freq[0])

        if cv.size > self.size:
            self.potential = resize_array(cv.potential, i_f - i_s)
            self.potential_units = cv.potential_units
            self.current = resize_array(cv.current, i_f - i_s)
            self.current_units = cv.current_units
            self.charge = resize_array(cv.charge, i_f - i_s)
            self.charge_units = cv.charge_units

        elif cv.size == self.size:
            self.potential = cv.potential
            self.potential_units = cv.potential_units
            self.current = cv.current
            self.current_units = cv.current_units
            self.charge = cv.charge
            self.charge_units = cv.charge_units

        elif cv.size < self.size:
            self.potential = cv.potential
            self.potential_units = cv.potential_units
            self.current = cv.current
            self.current_units = cv.current_units
            self.charge = cv.charge
            self.charge_units = cv.charge_units
            self.time = resize_array(self.time,cv.size)
            self.freq = resize_array(self.freq,cv.size)
            self.res = resize_array(self.res,cv.size)

        if verify:
            ax[0].plot(self.time, self.freq, label='pros')
            ax[1].plot(self.time, self.potential, label='pros')
            ax[0].legend();ax[1].legend()

    def clip_data(self,i_s=0,i_f='end', rezero=False):
        '''Remove data points, who's index is not between i_s and i_f'''

        if not self.is_valid(): return print('Error')

        if i_f == 'end':
            i_f = self.size
        self.time = self.time[i_s:i_f]
        self.freq = self.freq[i_s:i_f]
        self.res = self.res[i_s:i_f]

        try: self.potential = self.potential[i_s:i_f]
        except: pass
        try: self.current = self.current[i_s:i_f]
        except: pass
        try: self.charge = self.charge[i_s:i_f]
        except: pass

        if rezero:
            self.rezero()

    def plot(self, variable=None, i_s=0,i_f='end', xaxis='time'):
        '''
        Plot frequency and resistance vs time or potential

        Parameters
        ----------
        variable : str, optional
            Plot selected variable. The default is none.
        i_s : int, optional
            Start index. The default is 0.
        i_f : int, optional
            End index. The default is 'end'.
        xaxis : str, optional
            Independent variable, 'time' OR 'potential'. The default is 'time'.

        Returns
        -------
        FIG : TYPE
            DESCRIPTION.
        AX : TYPE
            DESCRIPTION.

        '''

        if variable:
            fig, ax = plt.subplots()
        else:
            fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True,figsize = (5,8),tight_layout=False)
            fig.subplots_adjust(hspace=0)
            text_figure(self.label, fig=fig, height=1)

        if i_f == 'end':
            i_f = self.size

        if xaxis == 'time':
            if variable:
                data = getattr(self, variable)
                ax.plot(self.time[i_s:i_f],data[i_s:i_f])
                ax.set_ylabel(getattr(self, variable + '_label'))
                ax.set_xlabel(self.time_label)

            else:
                ax[0].plot(self.time[i_s:i_f],self.freq[i_s:i_f])
                ax[1].plot(self.time[i_s:i_f],self.res[i_s:i_f])
                ax[0].set_ylabel(self.freq_label)
                ax[1].set_ylabel(self.res_label)
                ax[1].set_xlabel(self.time_label)

        elif xaxis == 'potential':
            if variable:
                data = getattr(self, variable)
                ax.plot(self.potential[i_s:i_f],data[i_s:i_f])
                ax.set_ylabel(getattr(self, variable + '_label'))
                ax.set_xlabel(self.potential_label)


            else:
                ax[0].plot(self.potential[i_s:i_f],self.freq[i_s:i_f])
                ax[1].plot(self.potential[i_s:i_f],self.res[i_s:i_f])
                ax[0].set_ylabel(self.freq_label)
                ax[1].set_ylabel(self.res_label)
                ax[1].set_xlabel(self.potential_label)

        return fig,ax

    def mass_to_charge_cont(self, npts_to_avg = 50):
        '''
        Calculate the mass [g] to charge [C] ratio based on a moving difference of both quantities.

        The units can be scaled by any factor, as long as the same factor is used for both mass and charge units,
        since their ratio is used in the calculation.

        Parameters
        ----------
        npts_to_avg : int, optional
            Point spacing used for calculating moving difference. The default is 50.

        Returns
        -------
        m2c_cont: ndarray
            Mass to charge ratio [g/mol] based on moving difference.
        slice_name: slice
            Use for plotting (usage: self.attr[slice_name]) since m2c_cont.size < self.size

        '''
        try:
            if type(self.charge) != np.ndarray:
                print('charge is not initialized')
                return None, None
            if self.charge.size != self.freq.size: # freq has same size as mass
                print('charge and frequency arrays are differnet sizes')
                return None, None

            if not ((self.charge_units == '$\mu$C/cm$^2$' and self.freq_units == 'Hz') or
                    (self.charge_units == 'mC/cm$^2$' and self.freq_units == 'kHz')):
                print('\n^^^^^^^^^^^^^^^^^\nCheck units!!!')
        except:
            print('charge is not initialized')
            return None, None

        diff_mass_cont = np.zeros(self.freq.size - npts_to_avg) #differential mass using moving difference
        diff_charge_cont = np.zeros(self.freq.size - npts_to_avg) #differential charge using moving difference
        for i in range(self.freq.size - npts_to_avg):
            diff_mass_cont[i] = self.mass[i + npts_to_avg] - self.mass[i]
            diff_charge_cont[i] = self.charge[i + npts_to_avg] - self.charge[i]

        m2c_cont = -diff_mass_cont * 96485 / diff_charge_cont
        slice_name = slice(round(npts_to_avg/2),-round(npts_to_avg/2))
        return m2c_cont, slice_name

    def smooth(self, window_size=21, order=3, deriv=0):
        if not self.is_valid(): return print('Error')
        self.time = savitzky_golay(self.time, window_size, order, deriv)
        self.freq = savitzky_golay(self.freq, window_size, order, deriv)
        self.res = savitzky_golay(self.res, window_size, order, deriv)

    def average_points(self,n):
        if not self.is_valid(): return print('Error')
        self.time = average(self.time, n)
        self.freq = average(self.freq, n)
        self.res = average(self.res, n)

    def scale_time(self, factor):
        self.time *= factor

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

    def scale_freq(self, factor):
        self.freq *= factor

        #Update freq_units
        qty = self.freq_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.freq_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.freq_units = prefix + qty[1]
            except: print('error')

    def scale_res(self, factor):
        self.res *= factor

        #Update res_units
        qty = self.res_units.split(' ')
        if len(qty) == 1:# no prefix
            prefix = '{:.3e} '.format(1/factor)
            self.res_units = prefix + qty[0]

        if len(qty) == 2:# numeric prefix
            try:
                value = float(qty[0])
                prefix = '{:.3e} '.format(value/factor)
                self.res_units = prefix + qty[1]
            except: print('error')

    def scale_potential(self, factor):
        try:
            self.potential *= factor
        except:
            print('Potential data does not exist')

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
        try:
            self.current *= factor
        except:
            print('Current data does not exist')

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
        except:
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

    def set_freq_units(self,units):
        '''
        1 --> kHz
        '''
        if type(units) == str:
            self.freq_units = units
        else:
            if units == 1:
                self.freq_units = 'kHz'
            else:
                print('Error!')

    def set_res_units(self,units):
        '''
        1 --> kΩ
        '''
        if type(units) == str:
            self.res_units = units
        else:
            if units == 1:
                self.res_units = 'k$\Omega$'
            else:
                print('Error!')

    def set_potential_units(self,units):
        '''
        1 --> V vs Li/Li⁺
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
        1 --> mA/cm²
        2 --> µA/cm²
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
        1 --> mC/cm²
        2 --> µC/cm²
        3 --> C/cm²
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

    def shift_time(self, shift):
        self.time += shift

    def shift_freq(self, shift):
        self.freq += shift

    def shift_res(self, shift):
        self.res += shift

    def shift_potential(self, shift):
        try:
            self.potential += shift

        except:
            print('Potential data does not exist')

    def shift_current(self, shift):
        try:
            self.current += shift

        except:
            print('Current data does not exist')

    def shift_charge(self, shift):
        try:
            self.charge += shift
        except:
            print('Charge data does not exist')

    def is_valid(self):
        '''
        Determine whether the eqcm instance has valid attributes.

        Returns
        -------
        bool
            True if the instance is valid.

        '''

        try:
            if (type(self.time) == type(self.freq) == type(self.res) == np.ndarray):
                return True
            else:
                print('Attribute type is not array')
                return False

        except AttributeError:
            print('Attribute is not defined')
            return False

    def rezero(self):
        self.time -= self.time[0]
        self.freq -= self.freq[0]
        try: self.charge -= self.charge[0]
        except TypeError: pass
