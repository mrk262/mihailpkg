# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:27:44 2021

@author: Mihail
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib import image as mpimg
from PIL import Image
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.simpledialog as sd
import tkinter.scrolledtext as st
import warnings
import os
import threading
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")
# warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

'''
Store a file's numeric data as an Array in a dict (data) with the filename as the key.
Store a file's column labels as a list of str in a separate dict (label) with the same filename as key.
'''
data = {'label':{}}
label = data['label']

def main():
    ENTRY_FONT = ('Times 14')
    LABEL_FONT = ('Times 14')
    MAIN_PAD = 2
    MAIN_WIDTH = 15
    MAIN_LABEL_COLOR = 'blue'


    """Load files into data dict and visualize the data with user interaction."""
    root = tk.Tk()                                                                #root widget
    root.title('Plot this shit')
    plt.style.use('mystyle')                                                        #style preferences for plotting
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    '''
    Each time data file(s) are loaded from a directory a new frame is created that
    contains the widgets which support interacting with the data. Directory names
    are displayed as a Label, filenames are displayed as elements in a Listbox. The following
    lists store the filemanes and references to those widgets
    '''
    full_filenames_list = []
    filenames = []
    directory_title_Entry_list = []
    Frame_list = []
    Scrollbar_list = []
    Listbox_list = []



    '''Store references to all the figures and axes that have been plotted'''
    fig_list = []
    axes_list = []
    SECM_plot_ref_list = []                                     #each element is a tuple fig,ax,CB

#%%--------------------------------main window commands------------------------

    def selectfiles():
        """Allow user to select data files from a directory, then display filenames for interactive plotting."""
        filenameslist = fd.askopenfilenames(filetypes = [('All files','*.*'),
                                                      ('Text files','*.txt'),
                                                      ('CSV files','*.csv'),
                                                      ('FuelCell3 files','*.fcd'),
                                                      ('NPY file','*.npy'),
                                                      ('MLG file','*.mlg')])
        data_folder = filenameslist[0].split('/')[-2]

        Frame_list.append(tk.Frame(main_app_files.secondFrame))
        Frame_list[-1].pack(fill='both', expand=True)
        Scrollbar_list.append(tk.Scrollbar(Frame_list[-1]))
        Scrollbar_list[-1].pack(side='right',fill='y', expand=True)
        directory_title_Entry_list.append(tk.Entry(Frame_list[-1], width = 40))
        directory_title_Entry_list[-1].original_title = data_folder
        directory_title_Entry_list[-1].insert(0,data_folder)
        directory_title_Entry_list[-1].pack(fill='both', expand=True)

        Listbox_list.append(MenuedListbox(Frame_list[-1],
                                          yscrollcommand = Scrollbar_list[-1].set,
                                          selectmode = 'multiple',
                                          exportselection = 0,
                                          width = 60))
        Listbox_list[-1].pack(fill='both', expand=True)
        Listbox_list[-1].bind('<Double-Button-1>',print_labels)


        Listbox_list[-1].bind('<Button-3>',lambda event: event.widget.popup_menu(event=event))

        for i,full_filename in enumerate(filenameslist):
            full_filenames_list.append(full_filename)
            if '.' == full_filename[-4]:
                filename = full_filename.split('/')[-1][:-4]
                if filename in filenames:
                    filename = filename + '$'
                filenames.append(filename)
            else:
                filename = full_filename.split('/')[-1]
                if filename in filenames:
                    filename = filename + '$'
                filenames.append(filename)
            Listbox_list[-1].insert("end",filenames[-1])
        Scrollbar_list[-1].config(command = Listbox_list[-1].yview)
        main_app_files.listboxContainer.update_idletasks()
        main_app_files.listboxContainer.configure(scrollregion= main_app_files.listboxContainer.bbox('all'))

    def plot_curve(ax = None, image=False):
        """Plot the user selected data files and load data into the dict."""
        if main_app_controls.overlay.get() and ax==None and image == False:                                                           #check if we want to overlay the selected data or plot it in a new graph
            fig,ax = plt.subplots()
            fig_list.append(fig)
            axes_list.append(ax)
            line_plot_window = LinePlotWindow(root,fig)
            line_plot_window.title('fig ' + str(fig.number - 1))
        k = -1                                                                      #tracks the index of file in file_names
        for Listbox_num, Listbox_reference in enumerate(Listbox_list):
            for data_num in range(Listbox_reference.size()):
                k +=1

                if data_num in Listbox_reference.curselection():                    #only plot user selected data

                    if image:
                        img = mpimg.imread(full_filenames_list[k])
                        fig, ax = plt.subplots()
                        ax.axis('off')
                        image = fig.figimage(img, resize=True)

                        fig_list.append(fig)
                        axes_list.append(ax)
                        secm_window = SecmWindow(root,fig)
                        secm_window.resizable(False,False)
                        secm_window.title('fig ' + str(fig.number - 1))

                    else:

                        if not main_app_controls.overlay.get() and ax == None:                                           #if not overlaying data, make a new figure for each data file
                            fig,ax = plt.subplots()
                            fig_list.append(fig)
                            axes_list.append(ax)
                            line_plot_window = LinePlotWindow(root,fig)
                            line_plot_window.title('fig ' + str(fig.number - 1))


                        x=int(main_app_controls.columns_to_plot.get().split(',')[0])                      #user selected data columns to plot based on their index
                        y=int(main_app_controls.columns_to_plot.get().split(',')[1])

                        data_array = parse_file(filenames[k],full_filenames_list[k])
                        ax.plot(data_array[:,x],
                                data_array[:,y],
                                label = filenames[k])
                        ax.legend()
                        L = ax.get_legend()
                        L.set_draggable(True)
                        try:
                            format_data_labels(filenames[k])
                            ax.set_xlabel(label[filenames[k]][x])
                            ax.set_ylabel(label[filenames[k]][y])
                        except:
                            pass
        ax.get_figure().canvas.draw()
        ax.get_figure().canvas.flush_events()
        deselect()



    def plot_2Dmap():
        """Plot a heat map from a 3 column Array of x,y,z data points."""
        string = sd.askstring('title','input bulk steady state curent(s) (comma separated) for normalization\n')
        iss_list = string.split(',')
        k = -1                                                                      #tracks the index of file in file_names
        n= -1                                                                       #tracks number of plots made
        for Listbox_num, Listbox_reference in enumerate(Listbox_list):
            for data_num in range(Listbox_reference.size()):
                k +=1
                if data_num in Listbox_reference.curselection():
                    n+=1

                    fig,ax = plt.subplots()
                    fig_list.append(fig)
                    axes_list.append(ax)
                    secm_window = SecmWindow(root,fig)
                    ax.set_xlabel('Distance / $\mu$m')
                    ax.set_ylabel('Distance / $\mu$m')

                    data_array = parse_file(filenames[k],full_filenames_list[k])
                    try:
                        iss = float(iss_list[n])
                    except:
                        try:
                            iss = float(iss_list[0])
                        except:
                            iss = 1e-9
                    try:
                        X,Y,Z = data_array
                        Z = Z / iss
                    except:

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
                        Z[:z.size] = z / iss
                        Z = Z.reshape(ypts,xpts)


                    secm_window.image = ax.imshow(Z, extent=[X.min(),X.max(),Y.min(),Y.max()], origin='lower', interpolation='bicubic', cmap='Spectral_r')
                    ax.set_aspect('equal')
                    secm_window.colorbar = fig.colorbar(secm_window.image,ax=ax, format = '%.2f')
                    fig.set_size_inches(6,5)

                    secm_window.title('fig ' + str(fig.number - 1) + ' | ' + filenames[k] + ' | ' + str(iss))
                    secm_window.ax = ax
                    secm_window.Z = Z

                    data[filenames[k]] = (X,Y,Z * iss)
                    SECM_plot_ref_list.append((fig,ax,secm_window.image,secm_window.colorbar))


        deselect()


    def parse_file(filename,full_filename):
        """Parse the data file for numeric data and append data array to the dictionary."""
        try: #if the file has been loaded previously, get it from the data dictionary
            data_array = data[filename]
            return data_array
        except KeyError:    #parse the file for the data, removing header and identifying data labels

            with open(full_filename, encoding='utf8') as file:
                text_file = file.readlines()

            '''Stores the strictly numerical data portion of the text file.
            Assumes the data structure is a N x M matrix of numbers with
            rows separated by '\n' and columns separated by user defined delimiter.
            The file may have a header and a footer'''

            numeric_data = []
            delim = main_app_controls.delimiter.get()

            for i in range(1,100):
                '''find the number of data columns by checking rows for numerical data,
                starting from the end of the file and going up.
                The footer is likely absent or small, errors are less likely then
                looping through the header first'''
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
                    if len(element) == 0: continue
                    if element[0] == '#': continue                  #comment character
                    try:                                            #if the row contains the correct number of elements and these are
                        numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
                        columns_list[j] = numeric_element
                        if j == num_col - 1:
                            numeric_data.append(columns_list)
                    except:
                        if len(columns_list) == num_col and j == 0: #if the (text) row contains the same number of elements as a data row,
                            label[filename] = columns_list          #it must be a label describing the data
                        continue
            data_array = np.array(numeric_data)
            data[filename] = data_array                         #add the data to the dictionary
            if filename not in label:
                label[filename] = list(range(num_col))
            return data_array


    def legend_OFF(event = None, ax=None):
        if ax:
            legend = ax.get_legend()
            if legend != None:
                legend.remove()
                ax.get_figure().canvas.draw()

        else:
            """Hide legend on all displayed figures."""
            for i,ax in enumerate(axes_list):
                legend = ax.get_legend()
                if legend == None:                                          #if the legend was removed from previous call, turn it back on
                    continue
                else:
                    legend.remove()
                fig_list[i].canvas.draw()                                   #update the change on the figure

    def legend_ON(event = None, ax=None):
        if ax:
            legend = ax.get_legend()
            if legend == None:
                legend = ax.legend()
                legend.set_draggable(True)
                ax.get_figure().canvas.draw()
        else:
            """Show legend on all displayed figures."""
            for i,ax in enumerate(axes_list):
                legend = ax.get_legend()
                if legend == None:                                          #if the legend was removed from previous call, turn it back on
                    legend = ax.legend()
                    legend.set_draggable(True)
                else:
                    continue
                fig_list[i].canvas.draw()                                   #update the change on the figure

    def deselect():
        """Deselect all active lines in all listboxes."""
        for Listbox_reference in Listbox_list:
            Listbox_reference.selection_clear(0,'end')

    def print_labels(event = None):
        """Print all the column labels of double-clicked file and plot only specified columns."""
        for l in Listbox_list:
            for i in l.curselection():
                selected_file = l.get(i)
                plot_curve()
                print('-'*20)
                print(selected_file,'\t','\n')
                column_labels = label[selected_file]
                for i,col_lab in enumerate(column_labels):
                    print(i,col_lab)
                print('-'*20,'\n')

#%%--------------------------------line plot menu commands---------------------

    def change_axis(ax):

        def update():
            ax.set_xlabel(xlabel.get())
            ax.set_ylabel(ylabel.get())

            x_scale = float(xscale_entry.get())
            y_scale = float(yscale_entry.get())

            x_shift = float(xshift_entry.get())
            y_shift = float(yshift_entry.get())

            ax.x_scale *= x_scale
            ax.y_scale *= y_scale

            ax.x_shift += x_shift
            ax.y_shift += y_shift

            for line in ax.lines:
                line.set_xdata((line.get_xdata() + x_shift)*x_scale)
                line.set_ydata((line.get_ydata() + y_shift)*y_scale)

            xscale_entry.delete(0, tk.END)
            xscale_entry.insert(0,'{:.2e}'.format(1.))
            yscale_entry.delete(0, tk.END)
            yscale_entry.insert(0,'{:.2e}'.format(1.))
            xshift_entry.delete(0, tk.END)
            xshift_entry.insert(0,'{:.2e}'.format(0.))
            yshift_entry.delete(0, tk.END)
            yshift_entry.insert(0,'{:.2e}'.format(0.))

            ax.relim()
            ax.autoscale()
            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()

        def reset():

            for line in ax.lines:
                line.set_xdata((line.get_xdata() - ax.x_shift)/ax.x_scale)
                line.set_ydata((line.get_ydata() - ax.y_shift)/ax.y_scale)
            ax.x_scale = 1.
            ax.y_scale = 1.
            ax.x_shift = 0.
            ax.y_shift = 0.

            ax.relim()
            ax.autoscale()

        win = tk.Toplevel(root)

        if not hasattr(ax, "x_scale"):
            ax.x_scale = 1.
        if not hasattr(ax, "y_scale"):
            ax.y_scale = 1.

        if not hasattr(ax, "x_shift"):
            ax.x_shift = 0.
        if not hasattr(ax, "y_shift"):
            ax.y_shift = 0.

        tk.Label(win,text = 'Set x label',font=LABEL_FONT, width=20).grid(row=0, column=0, columnspan=1, pady=5)
        xlabel = tk.Entry(win, font=ENTRY_FONT, width=20)
        xlabel.insert(0,ax.get_xlabel())
        xlabel.grid(row=1, column=0, columnspan=1, pady=5)

        tk.Label(win,text = 'Set y label',font=LABEL_FONT, width=20).grid(row=2, column=0, columnspan=1, pady=5)
        ylabel = tk.Entry(win, font=ENTRY_FONT, width=20)
        ylabel.insert(0,ax.get_ylabel())
        ylabel.grid(row=3, column=0, columnspan=1, pady=5)

        tk.Label(win,text = 'Scale x axis',font=LABEL_FONT, width=20).grid(row=0, column=1, columnspan=1, pady=5)
        xscale_entry = tk.Entry(win, font=ENTRY_FONT, width=20)
        xscale_entry.insert(0,'1')
        xscale_entry.grid(row=1, column=1, columnspan=1, pady=5)

        tk.Label(win,text = 'Scale y axis',font=LABEL_FONT, width=20).grid(row=2, column=1, columnspan=1, pady=5)
        yscale_entry = tk.Entry(win, font=ENTRY_FONT, width=20)
        yscale_entry.insert(0,'1')
        yscale_entry.grid(row=3, column=1, columnspan=1, pady=5)

        tk.Label(win,text = 'Shift x axis',font=LABEL_FONT, width=20).grid(row=0, column=2, columnspan=1, pady=5)
        xshift_entry = tk.Entry(win, font=ENTRY_FONT, width=20)
        xshift_entry.insert(0,'0')
        xshift_entry.grid(row=1, column=2, columnspan=1, pady=5)

        tk.Label(win,text = 'Shift y axis',font=LABEL_FONT, width=20).grid(row=2, column=2, columnspan=1, pady=5)
        yshift_entry = tk.Entry(win, font=ENTRY_FONT, width=20)
        yshift_entry.insert(0,'0')
        yshift_entry.grid(row=3, column=2, columnspan=1, pady=5)


        tk.Button(win, text="Update", command=update, padx=2, pady=2).grid(row=4, column=0, sticky='nesw', padx=5, pady=5)
        tk.Button(win, text="Reset", command=reset, padx=2, pady=2).grid(row=4, column=1, sticky='nesw', padx=5, pady=5)
        tk.Button(win, text="Close", command=win.destroy, padx=2, pady=2).grid(row=4, column=2, columnspan=2, sticky='nesw', padx=5, pady=5)


    def change_lines(ax):

        def update_legend():
            for line, label_entry in zip(ax.lines, label_entry_list):
                line.set_label(label_entry.get())
            for line, color_entry in zip(ax.lines, color_entry_list):
                color = color_entry.get()
                if ',' in color:
                    line.set_color(tuple([float(n) for n in color.strip('()').split(',')]))
                else:
                    line.set_color(color) # color is string
            for line, style_entry in zip(ax.lines, style_entry_list):
                line.set_linestyle(style_entry.get())
            ax.legend()
            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()

        def set_line_colors():

            def set_colors():
                cmap = plt.get_cmap(cmap_Entry.get())
                color_num_list = np.linspace(0,1,len(ax.lines))

                for color_num, line, entry in zip(color_num_list, ax.lines, color_entry_list):
                    color = cmap(color_num)
                    line.set_color(color)
                    entry.delete(0,tk.END)
                    entry.insert(0,str(color))

                ax.legend()
                ax.get_figure().canvas.draw()
                ax.get_figure().canvas.flush_events()
                sub_win.destroy()

            sub_win = tk.Toplevel(root)
            sub_win.grab_set()
            tk.Label(sub_win,text = 'Choose colormap', fg='red', font=LABEL_FONT, width=20).grid(row=0, column=0, columnspan=2, pady=5)
            cmap_Entry = tk.Entry(sub_win, font=ENTRY_FONT, width=20)
            cmap_Entry.insert(0,'jet')
            cmap_Entry.grid(row=1, column=0, columnspan=2, pady=5)
            tk.Button(sub_win, text="Set", command=set_colors, padx=5, pady=5).grid(row=2, column=0, sticky='nesw')
            tk.Button(sub_win, text="Close", command=sub_win.destroy, padx=5, pady=5).grid(row=2, column=1, sticky='nesw')



        win = tk.Toplevel(root)

        for line in ax.lines:
            if not hasattr(ax, "original_label"):
                line.original_label = line.get_label()

        label_entry_list = []
        color_entry_list = []
        style_entry_list = []

        for i,line in enumerate(ax.lines):
            tk.Label(win,text = 'Set label {}'.format(i), fg='red', font=LABEL_FONT, width=20).grid(row=2*i, column=0, pady=5)
            label_entry_list.append(tk.Entry(win, font=ENTRY_FONT, width=20))
            label_entry_list[i].insert(0,line.get_label())
            label_entry_list[i].grid(row=2*i+1, column=0, pady=5)

            tk.Label(win,text = 'Set color {}'.format(i), fg='red', font=LABEL_FONT, width=20).grid(row=2*i, column=1, pady=5)
            color_entry_list.append(tk.Entry(win, font=ENTRY_FONT, width=20))
            color_entry_list[i].insert(0,str(line.get_color()))
            color_entry_list[i].grid(row=2*i+1, column=1, pady=5)

            tk.Label(win,text = 'Set linestyle {}'.format(i), fg='red', font=LABEL_FONT, width=20).grid(row=2*i, column=2, pady=5)
            style_entry_list.append(tk.Entry(win, font=ENTRY_FONT, width=20))
            style_entry_list[i].insert(0,line.get_linestyle())
            style_entry_list[i].grid(row=2*i+1, column=2, pady=5)


        tk.Button(win, text="Update", command=update_legend, padx=5, pady=5).grid(row=2*i+2, column=0, sticky='nesw')
        tk.Button(win, text="Colormap", command=set_line_colors, padx=5, pady=5).grid(row=2*i+2, column=1, sticky='nesw')
        tk.Button(win, text="Close", command=win.destroy, padx=5, pady=5).grid(row=2*i+2, column=2, sticky='nesw')

    def trim_line(ax):
        def slider_changed(event):
            start_i = lower_index_slider.get()
            end_i = upper_index_slider.get()

            for line in ax.lines:
                line.set_xdata(line.original_x_data[start_i:end_i])
                line.set_ydata(line.original_y_data[start_i:end_i])

            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()

        win = tk.Toplevel(root)

        for line in ax.lines:
            if not hasattr(line, "original_x_data"):
                line.original_x_data = line.get_xdata()
            if not hasattr(line, "original_y_data"):
                line.original_y_data = line.get_ydata()


        max_index = min([line.get_xdata().size for line in ax.lines])

        lower_index_slider = tk.Scale(win, from_=0, to=max_index, orient=tk.HORIZONTAL, length=600, command=slider_changed)
        lower_index_slider.set(0)
        lower_index_slider.pack()

        upper_index_slider = tk.Scale(win, from_=0, to=max_index, orient=tk.HORIZONTAL, length=600, command=slider_changed)
        upper_index_slider.set(max_index)
        upper_index_slider.pack()

        tk.Button(win, text='Close', command=win.destroy).pack()

    def add_text(ax):

        def get_text():
            ax.user_text = text_entry.get()
            try:
                an = [child for child in ax.get_children() if isinstance(child, Annotation)][0]
                an.set_text(ax.user_text)
            except:
                ax.annotate(ax.user_text, xy=(0.5, 0.5), xycoords='axes fraction').draggable()
            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()

        win = tk.Toplevel(root)

        tk.Label(win,text = 'Add text',font=LABEL_FONT, width=20).grid(row=0, column=0, columnspan=2, pady=5)
        text_entry = tk.Entry(win, font=ENTRY_FONT, width=20)
        try:
            text_entry.insert(0,ax.user_text)
        except:
            text_entry.insert(0,'')
        text_entry.grid(row=1, column=0, columnspan=2, pady=5)

        tk.Button(win, text="Ok", command=get_text, padx=2, pady=2).grid(row=2, column=0, columnspan=1, sticky='nesw', padx=5, pady=5)
        tk.Button(win, text="Close", command=win.destroy, padx=2, pady=2).grid(row=2, column=1, columnspan=1, sticky='nesw', padx=5, pady=5)


    def save_data(ax):

        def save1():
            win.destroy()
            for line in ax.lines:
                if not hasattr(line, "original_label"):
                    line.original_label = line.get_label()
                data[line.original_label] = np.array(line.get_data()).T

        def save2():
            win.destroy()
            file = fd.asksaveasfile(filetypes = [('Multi line graph', '*.mlg'), ('All Files', '*.*')], defaultextension = [('Multi line graph', '*.mlg'), ('All Files', '*.*')])
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()
            file.write(x_label + '\t' + y_label + '\n')
            for line in ax.lines:
                label = line.get_label()
                file.write(label + '\n')
                xdata,ydata = line.get_data()
                for i in range(xdata.size):
                    file.write(str(xdata[i]) + '\t' + str(ydata[i]) + '\n')
            file.close()

        win = tk.Toplevel(root)
        win.grab_set()

        tk.Button(win, text="Save to instance", command=save1, padx=2, pady=2).grid(row=0, column=0, columnspan=1, sticky='nesw', padx=5, pady=5)
        tk.Button(win, text="Save to file", command=save2, padx=2, pady=2).grid(row=0, column=1, columnspan=1, sticky='nesw', padx=5, pady=5)
        tk.Button(win, text="Cancel", command=win.destroy, padx=2, pady=2).grid(row=0, column=2, columnspan=1, sticky='nesw', padx=5, pady=5)

#%%--------------------------------secm plot popup menu commands---------------

    def trim_secm(ax):
        def slider_changed(event):
            vmin = lower_index_slider.get()
            vmax = upper_index_slider.get()
            image.set_clim([vmin, vmax])

        win = tk.Toplevel(root)
        image = ax.get_images()[0]

        if not hasattr(image, "original_vmin"):
            image.original_vmin = image.get_clim()[0]
        if not hasattr(image, "original_vmax"):
            image.original_vmax = image.get_clim()[1]

        resolution = (image.original_vmax - image.original_vmin) / 50

        lower_index_slider = tk.Scale(win, from_= image.original_vmin, to = image.original_vmax, orient=tk.HORIZONTAL, length=600, command=slider_changed, resolution = resolution)
        lower_index_slider.set(image.original_vmin)
        lower_index_slider.pack()

        upper_index_slider = tk.Scale(win, from_= image.original_vmin, to=image.original_vmax, orient=tk.HORIZONTAL, length=600, command=slider_changed, resolution = resolution)
        upper_index_slider.set(image.original_vmax)
        upper_index_slider.pack()

        tk.Button(win, text='Close', command=win.destroy).pack()

#%%--------------------------------main window popup menu commands-------------




#%%

    def code_input():
        EDITOR_WIDTH = 10
        class EditorAppMenubar(tk.Menu):
            def __init__(self, parent, *args, **kwargs):
                tk.Menu.__init__(self, parent, *args, **kwargs)
                self.parent = parent
                self.parent.configure(menu = self)

                self.File_menu = tk.Menu(self)
                self.add_cascade(
                    label = 'File',
                    menu = self.File_menu)
                self.File_menu.add_command(
                    label = 'Save Data',
                    command = self.save_data)

                #   Insert menu
                self.Insert_menu = tk.Menu(self)
                self.add_cascade(
                    label = 'Insert',
                    menu = self.Insert_menu)
                self.Insert_menu.add_command(
                    label = 'Test',
                    command = self.insert_test)

            def insert_test(self):
                text_editor.insert(tk.END,"print('Test')")

            def save_data(self):
                import pickle
                file = fd.asksaveasfile(mode = 'wb', filetypes = [('Pickled Files', '*.pickle')], defaultextension = [('Pickled Files', '*.pickle')])
                pickle.dump(data,file)

        # Button functions

        def run():
            text = text_editor.get(1.0, tk.END)
            namespace = {'data':data, 'fig_list':fig_list, 'axes_list':axes_list, 'plt':plt, 'np':np}
            exec(text, namespace)
            if auto_refresh.get():
                for fig in fig_list:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        def save():
            text = text_editor.get(1.0,tk.END)
            files = [('Python Files', '*.py'),
                     ('Text Document', '*.txt'),
                     ('All Files', '*.*')]
            file = fd.asksaveasfile(filetypes = files, defaultextension = files)
            file.write(text)
            file.close()

        new_win = tk.Toplevel(root)
        new_win.bind('<Control-Return>', lambda event: run())
        auto_refresh = tk.IntVar()

        # Parent widget for the buttons
        buttons_frame = tk.Frame(new_win)#, relief=tk.RAISED, bd=2)
        buttons_frame.grid(row=0, column=0, sticky='NS')

        spacer = tk.Label(buttons_frame, text='   ', width=EDITOR_WIDTH)
        spacer.grid(row=0, column=0, padx=10, pady=MAIN_PAD)

        btn_run = tk.Button(buttons_frame, text='Run', width=EDITOR_WIDTH, command=run)
        btn_run.grid(row=1, column=0, padx=10, pady=MAIN_PAD)

        btn_File = tk.Button(buttons_frame, text='Save', width=EDITOR_WIDTH, command=save)
        btn_File.grid(row=2, column=0, padx=10, pady=MAIN_PAD)

        btn_Folder = tk.Button(buttons_frame, text='Close', width=EDITOR_WIDTH, command=new_win.destroy)
        btn_Folder.grid(row=3, column=0, padx=10, pady=MAIN_PAD)

        srefreshButton = tk.Checkbutton(buttons_frame, text = 'Refresh', variable = auto_refresh)
        srefreshButton.select()
        srefreshButton.grid(row=4,column = 0, padx=10, pady=MAIN_PAD)

        # Parent widget for the text editor
        text_Frame = tk.LabelFrame(new_win, text="Code", padx=5, pady=5)
        text_Frame.grid(row=0, column=1, rowspan=5, padx=MAIN_PAD, pady=MAIN_PAD, sticky='NSEW')

        new_win.columnconfigure(1, weight=1)
        new_win.rowconfigure(0, weight=1)

        text_Frame.rowconfigure(0, weight=1)
        text_Frame.columnconfigure(0, weight=1)

        # Create the textbox
        text_editor = st.ScrolledText(text_Frame)
        text_editor.insert('1.0',
                           '#from mihailpkg import cv_processing as cp\n' +
                           '#ax = axes_list[0]\n')
        text_editor.grid(row=0, column=0,   sticky='NSEW')

        # Create menu bar
        manubar = EditorAppMenubar(new_win)


    def replot_secm(fig):
        pass

    def format_data_labels(filename):
        for i,col_label in enumerate(label[filename]):
            if col_label == 'Time (Sec)':
                label[filename][i] = 'Time / Sec'
            if col_label == 'I (A)':
                label[filename][i] = 'Current / Amps'
            if col_label == 'I (mA/cmÂ²)':
                label[filename][i] = 'Current density / mA/cm$^2$'
            if col_label == 'Power (Watts)':
                label[filename][i] = 'Power / Watts'
            if col_label == 'Power (mW/cmÂ²)':
                label[filename][i] = 'Power density / mW/cm$^2$'
            if col_label == 'E_Stack (V)':
                label[filename][i] = 'Cell potential / V'
            if col_label == 'Temp (C)':
                label[filename][i] = 'Cell temp / $^o$C'
            if col_label == 'HFR (mOhm)':
                label[filename][i] = 'HFR / mOhm'
            if col_label == 'Z_Real (Ohm)':
                label[filename][i] = 'Z$_{Real}$ / $\Omega$'
            if col_label == 'Z_Imag (Ohm)':
                label[filename][i] = 'Z$_{Imag}$ / $\Omega$'
            # if col_label == 'Time (Sec)':
            #     label[filename][i] = 'Time / sec'
            # if col_label == 'Time (Sec)':
            #     label[filename][i] = 'Time / sec'
            # if col_label == 'Time (Sec)':
            #     label[filename][i] = 'Time / sec'
            # if col_label == 'Time (Sec)':
            #     label[filename][i] = 'Time / sec'

#%%--------------------------------build the Main window-----------------------

    class MainAppMenubar(tk.Menu):
        '''
        Menubar in the root widget
        Initialized in MainAppControls Frame
        '''
        def __init__(self, parent, *args, **kwargs):
            tk.Menu.__init__(self, parent, *args, **kwargs)
            self.parent = parent

            self.parent.configure(menu = self)
            self.File_menu = tk.Menu(self)
            self.Help_menu = tk.Menu(self)

            self.File_menu.add_command(
                label = 'rename',
                command = self.rename_filename)
            self.File_menu.add_command(
                label = 'remove',
                command = self.remove_files)

            self.Help_menu.add_command(
                label = 'shortcuts',
                command = self.help_shortcuts)

            self.add_cascade(
                label = 'File',
                menu = self.File_menu)
            self.add_cascade(
                label = 'Help',
                menu = self.Help_menu)



        def rename_filename(self):
            for n,l in enumerate(Listbox_list):
                for i in l.curselection():
                    selected_filename = l.get(i)
                    new_filename = sd.askstring(title ='',
                                                             prompt = 'Rename file below',
                                                             initialvalue = selected_filename)
                    if new_filename == None: return
                    l.delete(i)
                    l.insert(i, new_filename)
                    for m,file in enumerate(full_filenames_list):
                        file_structure = file.split('/')
                        if (selected_filename.strip('$') == file_structure[-1][:-4]) and (
                                directory_title_Entry_list[n].original_title == file_structure[-2]):

                            directory = ''
                            for folder in file_structure[:-1]:
                                directory += folder + '/'
                            os.rename(file, directory + new_filename + file[-4:])
                            full_filenames_list[m] = directory + new_filename + file[-4:]


        def remove_files(self):
            def delete_files():
                confirm_win.destroy()
                for n,l in enumerate(Listbox_list):
                    for i,line_number in enumerate(l.curselection()):
                        filename = l.get(line_number - i)

                        for j,full_name in enumerate(full_filenames_list):
                            if filename in full_name:
                                del full_filenames_list[j]


                        try: del filenames[filenames.index(filename)]
                        except: pass
                        try: del data[filename]
                        except: pass
                        l.delete(line_number - i)

            confirm_win = tk.Toplevel(root)
            confirm_win.grab_set()
            tk.Label(confirm_win, text='Delete selected files?',font=LABEL_FONT, width=40, height=5).grid(row = 0, column=0, columnspan=2)
            tk.Button(confirm_win, text='Yes', command=delete_files).grid(row = 1, column=0, columnspan=1, sticky="nsew", pady=5, padx=5)
            tk.Button(confirm_win, text='No', command=lambda:confirm_win.destroy()).grid(row = 1, column=1, columnspan=1, sticky="nsew", pady=5, padx=5)


        def help_shortcuts(self):
            new_win = tk.Toplevel()

            PADX = 15
            PADY = 10
            WIDTH_KEY = 15
            WITH_DESC = 20
            FONT = ('Times 12')

            new_win.number_of_shortcuts = 0

            def add_desc(key, func):

                keyLabel = tk.Label(new_win, text=key, width=WIDTH_KEY, font=FONT)
                funcLabel = tk.Label(new_win, text=func, width=WITH_DESC, font=FONT)
                keyLabel.grid(row=new_win.number_of_shortcuts, column=0, padx=PADX, pady=PADY)
                funcLabel.grid(row=new_win.number_of_shortcuts, column=1, padx=PADX, pady=PADY)
                new_win.number_of_shortcuts +=1

            add_desc('a', 'Select all')
            add_desc('Shift + a', 'Deselect all')
            add_desc('r', 'Refresh plot')
            add_desc('Ctr + Enter', 'Run code')
            add_desc('l', 'Legend on')
            add_desc('Shift + l', 'Legend off')
            add_desc('Ctrl + c', 'Copy filenames')
            #add_desc('r', 'Refresh plot')


    class MainAppControls(tk.Frame):
        def __init__(self, parent, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, **kwargs)

            self.main_app_menubar = MainAppMenubar(root)

            self.parent = parent
            self.delimiter = tk.StringVar()
            self.columns_to_plot = tk.StringVar()
            self.overlay = tk.IntVar()

            self.selectButton = tk.Button(self,text='Open Files',command = selectfiles, width=MAIN_WIDTH)
            self.selectButton.grid(row = 0, column = 0, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

            self.overlayButton = tk.Checkbutton(self,text = 'Overlay',variable = self.overlay)
            self.overlayButton.select()
            self.overlayButton.grid(row=0,column = 1,columnspan = 1, padx=MAIN_PAD, pady=MAIN_PAD)

            self.delimiterLabel = tk.Label(self,text = 'Delimiter:', width=MAIN_WIDTH, fg=MAIN_LABEL_COLOR)
            self.delimiterLabel.grid(row = 1, column = 0, padx=MAIN_PAD, pady=MAIN_PAD)

            self.delimiterBox = tk.Entry(self,textvariable = self.delimiter, width=10)
            self.delimiterBox.insert(0,'\t')
            self.delimiterBox.grid(row=1,column=1,columnspan = 1)

            self.cols_to_plotLabel = tk.Label(self, text='Cols to plot (x,y):', width=MAIN_WIDTH, fg=MAIN_LABEL_COLOR)
            self.cols_to_plotLabel.grid(row=2,column=0,columnspan = 1, padx=MAIN_PAD, pady=MAIN_PAD)

            self.cols_to_plotBox = tk.Entry(self,textvariable = self.columns_to_plot, width=10)
            self.cols_to_plotBox.insert(0,'0,1')
            self.cols_to_plotBox.grid(row=2,column=1,columnspan = 1, padx=MAIN_PAD, pady=MAIN_PAD)

            self.plotButton = tk.Button(self,text='Plot Line',command = plot_curve, width=MAIN_WIDTH)
            self.plotButton.grid(row = 3, column = 0,columnspan = 1, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

            self.plot2D_Button = tk.Button(self,text='Plot SECM',command = plot_2Dmap, width=MAIN_WIDTH)
            self.plot2D_Button.grid(row = 3, column = 1,columnspan = 1, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

            self.toggleLegend_Button = tk.Button(self,text='Plot Image', command = lambda: plot_curve(image=True))
            self.toggleLegend_Button.grid(row = 4, column = 0, columnspan = 1, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

            self.code_Button = tk.Button(self,text='Code',command = code_input, width=MAIN_WIDTH)
            self.code_Button.grid(row = 4, column = 1,columnspan = 1, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

            self.bind('<Control-o>', lambda event: selectfiles())
            self.focus_set()

    class MainAppFiles(tk.Frame):
        '''
        Container for the Listboxes which contain filenames.
        Each Listbox holds a list of filenames loaded from a single directory.
        '''

        def __init__(self, parent, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, **kwargs)
            self.parent = parent

            self.listboxContainer = tk.Canvas(self, bg="red")
            self.listboxContainer.pack(side='left', fill='both', expand=True)
            self.canvasScrollbar = tk.Scrollbar(self,command = self.listboxContainer.yview)
            self.canvasScrollbar.pack(side='right',fill='y')
            self.listboxContainer.configure(yscrollcommand = self.canvasScrollbar.set)
            self.listboxContainer.bind('<Configure>',lambda e: self.listboxContainer.configure(scrollregion= self.listboxContainer.bbox('all')))
            self.secondFrame = tk.Frame(self.listboxContainer)
            self.listboxContainer.create_window((0,0),window = self.secondFrame,anchor='nw')

#%%--------------------------------build the SECM window-----------------------

    class SecmWindow(tk.Toplevel):

        class AppRclickMenu(tk.Menu):
            def __init__(self, parent, ax, *args, **kwargs):
                tk.Menu.__init__(self, parent, *args, **kwargs)
                self.parent = parent

                self.add_command(
                    label = 'axis format',
                    command = lambda: change_axis(ax))
                self.add_command(
                    label = 'add text',
                    command = lambda: add_text(ax))
                self.add_command(
                    label = 'trim data',
                    command = lambda: trim_secm(ax))


        def __init__(self, parent, fig, *args, **kwargs):
            tk.Toplevel.__init__(self, parent, *args, **kwargs)

            self.fig = fig
            self.ax = fig.axes[0]

            self.app_rclick_menu = SecmWindow.AppRclickMenu(self,self.ax)

            self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="bottom", fill='both',expand=True)
            self.canvas.get_tk_widget().bind('<Button-3>', lambda x: self.popup_menu(event=x))

            self.toolbarFrame = tk.Frame(master=self)
            self.toolbarFrame.pack(side="top",fill='x',expand=False)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)

        def popup_menu(self, event = None):
            try:
                self.app_rclick_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.app_rclick_menu.grab_release()


#%%--------------------------------build the line plot window------------------

    class LinePlotWindow(tk.Toplevel):
        class AppRclickMenu(tk.Menu):
            def __init__(self, parent, ax, *args, **kwargs):
                tk.Menu.__init__(self, parent, *args, **kwargs)
                self.parent = parent

                self.add_command(
                    label = 'axis format',
                    command = lambda: change_axis(ax))
                self.add_command(
                    label = 'line format',
                    command = lambda: change_lines(ax))
                self.add_command(
                    label = 'trim data',
                    command = lambda: trim_line(ax))
                self.add_command(
                    label = 'add text',
                    command = lambda: add_text(ax))
                self.add_command(
                    label = 'save data',
                    command = lambda: save_data(ax))
                self.add_command(
                    label = 'add lines',
                    command = lambda: plot_curve(ax=ax))


        def __init__(self, parent, fig, *args, **kwargs):
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
            self.fig = fig
            self.ax = fig.axes[0]

            self.app_rclick_menu = LinePlotWindow.AppRclickMenu(self,self.ax)


            self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="bottom",fill='both',expand=True)
            self.canvas.get_tk_widget().bind('<Button-3>', self.popup_menu)
            self.canvas.get_tk_widget().bind('l', lambda x: legend_ON(ax = self.ax, event=x))
            self.canvas.get_tk_widget().bind('<Shift-L>', lambda x: legend_OFF(ax = self.ax, event=x))
            self.canvas.get_tk_widget().bind('r', self.refresh)


            self.toolbarFrame = tk.Frame(master=self)
            self.toolbarFrame.pack(side="top",fill='x',expand=False)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)

        def popup_menu(self, event=None):
            try:
                self.app_rclick_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.app_rclick_menu.grab_release()

        def refresh(self, event=None):
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    class MenuedListbox(tk.Listbox):
        def __init__(self, parent, *args, **kwargs):
            tk.Listbox.__init__(self, parent, *args, **kwargs)

            self.bind('a', lambda event: self.select_all())
            self.bind('<Shift-A>', lambda event: self.deselect())

            self.menu = tk.Menu(self)
            self.menu.parent = self

            self.menu.add_command(
                label = 'add',
                command = self.add_user_data)
            self.menu.add_command(
                label = 'sort',
                command = self.sort_labels)
            self.menu.add_command(
                    label = 'select all',
                    command = self.select_all)
            self.menu.add_command(
                    label = 'copy',
                    command = self.copy)

            self.bind('<Control-c>', lambda event: self.copy())

        def add_user_data(self):
            filenameslist = fd.askopenfilenames(filetypes = [('All files','*.*'),
                                                          ('Text files','*.txt'),
                                                          ('CSV files','*.csv'),
                                                          ('FuelCell3 files','*.fcd'),
                                                          ('NPY file','*.npy'),
                                                          ('MLG file','*.mlg')])
            data_folder = filenameslist[0].split('/')[-2]
            label_list = [Entry.original_title for Entry in directory_title_Entry_list]
            if label_list.index(data_folder) != Listbox_list.index(self):
                print('Wrong Directory')
                return
            for i,full_filename in enumerate(filenameslist):
                if full_filename in full_filenames_list: continue
                full_filenames_list.append(full_filename)
                if '.' == full_filename[-4]:
                    filename = full_filename.split('/')[-1][:-4]
                    if filename in filenames:
                        filename = filename + '$'
                    filenames.append(filename)
                else:
                    filename = full_filename.split('/')[-1]
                    if filename in filenames:
                        filename = filename + '$'
                    filenames.append(filename)
                self.insert("end",filenames[-1])

        def sort_labels(self):
            labels = list(self.get(0,'end'))
            labels.sort()
            self.delete(0,'end')
            for label in labels:
                self.insert('end',label)

        def select_all(self):
            self.select_set(0,tk.END)

        def deselect(self):
            self.selection_clear(0,'end')

        def copy(self):
            text = ''
            for i in self.curselection():
                filename = self.get(i)
                text += "'" + filename + "'" + '\n'
            root.clipboard_clear()
            root.clipboard_append(text)
            self.selection_clear(0,'end')

        def popup_menu(self, event):
            try:
                self.menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.menu.grab_release()


#%%--------------------------------Main Program--------------------------------

    main_app_controls = MainAppControls(root, bg='red')                                                     #holds widgets for user interaction
    main_app_controls.pack(side='top')

    main_app_files = MainAppFiles(root)
    main_app_files.pack(side='bottom', fill='both', expand=True)

    root.mainloop()

def start_app():
    t1 = threading.Thread(target=main)
    t1.start()
if __name__ == '__main__':
    main()
