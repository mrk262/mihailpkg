#########################################################################
# DAQmx Python - Producer/Consumer example
# Updated 10/19/2018
#
# Reads continuous samples from a single physical channel and writes
# them to a log file. Uses two parallel threads to perform
# DAQmx Read calls and data processing.
#
# Note: The number of samples per execution varies slightly since
# the task's start and stop times are specified in software.
#
# Input Type: Analog Voltage
# Dependencies: nidaqmx
#########################################################################

import numpy as np
import nidaqmx
from nidaqmx import stream_readers
import time
from timeit import default_timer as timer
import queue
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
import tkinter.filedialog as fd

plt.style.use('mystyle')
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.ioff()
DIFFERENTIAL = nidaqmx.constants.TerminalConfiguration.DIFF
CONTINUOUS = nidaqmx.constants.AcquisitionType.CONTINUOUS
CHANNELS = 4

def create_task():
    global task, buffer, out_file, task_reader

    task = nidaqmx.task.Task()
    task.ai_channels.add_ai_voltage_chan('Dev2/ai0', terminal_config=DIFFERENTIAL, min_val= -10, max_val= 10)
    task.ai_channels.add_ai_voltage_chan('Dev2/ai1', terminal_config=DIFFERENTIAL, min_val= -10, max_val= 10)
    task.ai_channels.add_ai_voltage_chan('Dev2/ai2', terminal_config=DIFFERENTIAL, min_val= -10, max_val= 10)
    task.ai_channels.add_ai_voltage_chan('Dev2/ai3', terminal_config=DIFFERENTIAL, min_val= -10, max_val= 10)
    task.timing.cfg_samp_clk_timing(rate=inputs_win.sample_rate,
                                    sample_mode=CONTINUOUS)
    buffer = np.zeros((CHANNELS, inputs_win.sample_size), dtype=np.float64)

    task_reader = stream_readers.AnalogMultiChannelReader(task.in_stream)

    filename = fd.asksaveasfilename()
    out_file = open(filename, 'w', encoding='utf8')
    header = 'Scan rate: {} mV/s\nTime / s\tPotential / V vs {}\tCurrent / {}\tFrequency / Hz\tResistance / Ω\n'.format(inputs_win.scan_rate, inputs_win.refernce_electrode, inputs_win.current_units)
    out_file.write(header)

# Reads any available samples from the DAQ buffer and places them on the queue.
# Runs for sample_duration seconds.
def producer_loop(q, task):
    start_time = time.time();
    while(time.time() - start_time < inputs_win.sample_duration):
        if stop: break
        task_reader.read_many_sample(buffer, number_of_samples_per_channel=inputs_win.sample_size, timeout= 10)
        q.put_nowait(buffer)
    task.stop()
    return

# Takes samples from the queue and writes them to LOG_FILE_PATH.
def consumer_loop(q, task, file):
    while(True):
        if stop: break
        try:
            temp = q.get(block=True, timeout=2)
        except:
            if (task.is_task_done()):
                return

        val0 = timer() - start                                                  # time
        val1 = np.average(temp[0])                                              # potential
        if inputs_win.electrode_area: #scale current by electrode area
            val2 = np.average(temp[1]) * inputs_win.current_converter / inputs_win.electrode_area
        else:
            val2 = np.average(temp[1]) * inputs_win.current_converter           # current
        val3 = np.average(temp[2]) * inputs_win.frequency_converter             # frequency
        val4 = 1e4*10**(-1*np.average(temp[3])/5) - 75                          # resistance

        file.write("{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n".format(val0,val1,val2,val3,val4))
        data[0].append(val0)
        data[1].append(val1)
        data[2].append(val2)
        data[3].append(val3)
        data[4].append(val4)
        q.task_done()

def plotting_loop():
    while cons.is_alive():
        i = len(data[0])
        App.ln0.set_xdata(data[1][:i])  # potential
        App.ln0.set_ydata(data[2][:i])  # current
        App.ln1.set_xdata(data[0][:i])  # time
        App.ln1.set_ydata(data[2][:i])  # current
        App.ln2.set_xdata(data[0][:i])  # time
        App.ln2.set_ydata(data[3][:i])  # frequency
        App.ln3.set_xdata(data[0][:i])  # time
        App.ln3.set_ydata(data[4][:i])  # resistance
        App.fig.canvas.draw()
        App.fig.canvas.flush_events()
        #plt.pause(0.05)

def run():

    global start, stop, data, cons
    stop = False
    # Start acquisition and threads
    q = queue.Queue()
    prod = threading.Thread(target=producer_loop, args=(q, task), daemon=True)
    cons = threading.Thread(target=consumer_loop, args=(q, task, out_file), daemon=True)
    data = [[],[],[],[],[]]
    task.start()
    prod.start()
    cons.start()
    start = timer()
    print("Task is running")
    plotting_loop()

    while(not task.is_task_done()):
        pass # Spin parent thread until task is done
    print("Task is done")

    while(cons.is_alive()):
        pass # Wait for consumer to finish

    print("Consumer finished")

    time.sleep(0.5)
    if stop: return
    out_file.close()
    print("Done!")
    reset()

def reset():
    global stop, out_file
    stop = True
    time.sleep(0.5)
    task.stop()
    out_file.close()
    filename = fd.asksaveasfilename()
    out_file = open(filename, 'w', encoding='utf8')
    header = 'Scan rate: {} mV/s, current converter: {}\nTime / s\tPotential / V vs {}\tCurrent / {}\tFrequency / Hz\tResistance / Ω\n'.format(inputs_win.scan_rate, inputs_win.current_converter, inputs_win.refernce_electrode, inputs_win.current_units)
    out_file.write(header)

def close():
    out_file.close()
    task.close()
    main_window.destroy()

class MainApp():

    def __init__(self, parent, *args, **kwargs):

        self.fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6.5), tight_layout = False)
        self.fig.subplots_adjust(wspace=0.15, hspace=0.18)
        ax0,ax1,ax2,ax3 = ax[0,0], ax[0,1], ax[1,0], ax[1,1],
        canvas = FigureCanvasTkAgg(self.fig, master=parent)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ln0 = ax0.plot([],[])[0]
        ax0.set_xlim(inputs_win.lower_potential_limit - 0.02,
                     inputs_win.upper_potential_limit + 0.02)
        ax0.set_xlabel('Potential / V vs {}'.format(inputs_win.refernce_electrode))
        ax0.set_ylim(-11 * inputs_win.current_converter,11 * inputs_win.current_converter)
        if inputs_win.electrode_area:
            inputs_win.current_units += '/cm²'
        ax0.set_ylabel('Current / {}'.format(inputs_win.current_units))

        self.ln1 = ax1.plot([],[])[0]
        ax1.set_xlim(0, inputs_win.sample_duration)
        ax1.set_xlabel('Time / s')
        ax1.set_ylim(-11 * inputs_win.current_converter,11 * inputs_win.current_converter)
        ax1.axes.yaxis.set_visible(False)

        self.ln2 = ax2.plot([],[])[0]
        ax2.set_xlim(0, inputs_win.sample_duration)
        ax2.set_xlabel('Time / s')
        ax2.set_ylim(-inputs_win.frequency_converter * 10, inputs_win.frequency_converter * 0.2)
        ax2.set_ylabel('Frequency / Hz')

        self.ln3 = ax3.plot([],[])[0]
        ax3.set_xlim(0, inputs_win.sample_duration)
        ax3.set_xlabel('Time / s')
        ax3.set_ylim(0, 900)
        ax3.set_ylabel('Resistance / Ω')

        self.Savebutton = tk.Button(master=parent, text="Close", command=close, width = 30)
        self.Savebutton.pack(side=tk.RIGHT)

        self.Stopbutton = tk.Button(master=parent, text="Reset", command=reset, width = 30)
        self.Stopbutton.pack(side=tk.RIGHT)

        self.Runbutton = tk.Button(master=parent, text="Run", command=run, width = 30)
        self.Runbutton.pack(side=tk.RIGHT)

class InputWindow(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, parent, *args, **kwargs)
        self.grab_set()

        tk.Label(self,text = 'Sample duration / s').grid(row=0, column=0, columnspan=2)
        self.sample_duration = tk.Entry(self)
        self.sample_duration.insert(0,'7200')
        self.sample_duration.grid(row=1, column=0, columnspan=2)

        tk.Label(self,text = 'Current units').grid(row=2, column=0, columnspan=2)
        self.current_units = tk.Entry(self)
        self.current_units.insert(0,'mA')
        self.current_units.grid(row=3, column=0, columnspan=2)

        tk.Label(self,text = 'Current converter').grid(row=4, column=0, columnspan=2)
        self.current_converter = tk.Entry(self)
        self.current_converter.insert(0,'1')
        self.current_converter.grid(row=5, column=0, columnspan=2)

        tk.Label(self,text = 'Electrode area').grid(row=6, column=0, columnspan=2)
        self.electrode_area = tk.Entry(self)
        self.electrode_area.insert(0,'0')
        self.electrode_area.grid(row=7, column=0, columnspan=2)

        tk.Label(self,text = 'Scan rate / mV/s').grid(row=8, column=0, columnspan=2)
        self.scan_rate = tk.Entry(self)
        self.scan_rate.insert(0,'20')
        self.scan_rate.grid(row=9, column=0, columnspan=2)

        tk.Label(self,text = 'Refernce electrode').grid(row=10, column=0, columnspan=2)
        self.refernce_electrode = tk.Entry(self)
        self.refernce_electrode.insert(0,'Li/Li⁺')
        self.refernce_electrode.grid(row=11, column=0, columnspan=2)

        tk.Label(self,text = 'Lower potential limit / V').grid(row=12, column=0, columnspan=2)
        self.lower_potential_limit = tk.Entry(self)
        self.lower_potential_limit.insert(0,'-1')
        self.lower_potential_limit.grid(row=13, column=0, columnspan=2)

        tk.Label(self,text = 'Upper potential limit / V').grid(row=14, column=0, columnspan=2)
        self.upper_potential_limit = tk.Entry(self)
        self.upper_potential_limit.insert(0,'1')
        self.upper_potential_limit.grid(row=15, column=0, columnspan=2)

        tk.Label(self,text = 'Frequency converter').grid(row=16, column=0, columnspan=2)
        self.frequency_converter = tk.Entry(self)
        self.frequency_converter.insert(0,'200')
        self.frequency_converter.grid(row=17, column=0, columnspan=2)

        tk.Label(self,text = 'Sample size').grid(row=18, column=0, columnspan=2)
        self.sample_size = tk.Entry(self)
        self.sample_size.insert(0,'50')
        self.sample_size.grid(row=19, column=0, columnspan=2)

        tk.Label(self,text = 'Sample rate').grid(row=20, column=0, columnspan=2)
        self.sample_rate = tk.Entry(self)
        self.sample_rate.insert(0,'3.1415e3')
        self.sample_rate.grid(row=21, column=0, columnspan=2)

        self.update = tk.Button(self, text="Continue", command=self.copy, padx=2, pady=2).grid(row=22, column=0, sticky='nesw')
        self.close = tk.Button(self, text="Close", command=parent.destroy, padx=2, pady=2).grid(row=22, column=1, sticky='nesw')
        self.protocol('WM_DELETE_WINDOW', parent.destroy)
        self.wait_window(self)

    def copy(self):
        self.sample_duration = float(self.sample_duration.get())
        self.current_units = self.current_units.get()
        self.current_converter = float(self.current_converter.get())
        self.electrode_area = float(self.electrode_area.get())
        self.scan_rate = float(self.scan_rate.get())
        self.refernce_electrode = self.refernce_electrode.get()
        self.lower_potential_limit = float(self.lower_potential_limit.get())
        self.upper_potential_limit = float(self.upper_potential_limit.get())
        self.frequency_converter = float(self.frequency_converter.get())
        self.sample_size = int(self.sample_size.get())
        self.sample_rate = float(self.sample_rate.get())
        self.destroy()


# Main program
if __name__ == "__main__":

    main_window = tk.Tk()
    main_window.wm_title('Lets do this science shit bruv')
    main_window.protocol('WM_DELETE_WINDOW', close)

    inputs_win = InputWindow(main_window)

    create_task()
    App = MainApp(main_window)

    tk.mainloop()
