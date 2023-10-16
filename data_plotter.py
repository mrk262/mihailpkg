# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:27:44 2021

@author: Mihail
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib import cm
from PIL import Image, ImageDraw, ImageTk, ImageOps
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import simpledialog
import warnings
import os
import sys, traceback
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

warnings.filterwarnings("ignore")

'''
Store a file's numeric data as an Array in a dict (data) with the filename as the key.
Store a file's column labels as a list of str in a separate dict (label) with the same filename as key.
'''
data = {'label':{}, 'metadata':{}}
label = data['label']
metadata = data['metadata']

def main():
    ENTRY_FONT = ('Times 14')
    LABEL_FONT = ('Times 14')
    MAIN_PAD = 2
    MAIN_WIDTH = 15
    MAIN_LABEL_COLOR = 'blue'


    """Load files into data dict and visualize the data with user interaction."""
    root = tk.Tk()                                                                #root widget
    root.title('Plot this shit')
    root.iconbitmap(r'metanoia.ico')
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
        filenameslist = filedialog.askopenfilenames(filetypes = [('All files','*.*'),
                                                      ('Text files','*.txt'),
                                                      ('CSV files','*.csv'),
                                                      ('FuelCell3 files','*.fcd'),
                                                      ('NPY file','*.npy'),
                                                      ('MLG file','*.mlg')])
        if filenameslist == '': return
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

    def plot_curve(ax = None, image_file=False):
        """Plot the user selected data files and load data into the dict."""
        if main_app_controls.overlay.get() and ax==None and image_file == False: # create new window for line plots, overlaying all
            fig,ax = plt.subplots()
            fig_list.append(fig)
            axes_list.append(ax)
            line_plot_window = LinePlotWindow(root,fig)
            line_plot_window.title('fig ' + str(fig.number - 1))

        for Listbox_num, Listbox_reference in enumerate(Listbox_list): # loop through all filenames and find selected ones
            for data_num in range(Listbox_reference.size()):
                if data_num in Listbox_reference.curselection():
                    filename = Listbox_reference.get(data_num)
                    full_filename = full_filenames_list[filenames.index(filename)]

                    if image_file:

                        image = Image.open(full_filename)
                        img_window = ImageWindow(root,image, filename=filename)

                    else:

                        if not main_app_controls.overlay.get() and ax == None:  #if not overlaying data, make a new figure for each data file
                            fig,ax = plt.subplots()
                            fig_list.append(fig)
                            axes_list.append(ax)
                            line_plot_window = LinePlotWindow(root,fig)
                            line_plot_window.title('fig ' + str(fig.number - 1))


                        x=int(main_app_controls.columns_to_plot.get().split(',')[0])                      #user selected data columns to plot based on their index
                        y=int(main_app_controls.columns_to_plot.get().split(',')[1])

                        data_array = parse_file(filename,full_filename)
                        ax.plot(data_array[:,x],
                                data_array[:,y],
                                label = filename)
                        ax.legend()
                        L = ax.get_legend()
                        L.set_draggable(True)
                        ax.get_figure().canvas.draw()
                        ax.get_figure().canvas.flush_events()


    def plot_2Dmap():
        """Plot a heat map from a 3 column Array of x,y,z data points."""
        string = simpledialog.askstring('title','input bulk steady state curent(s) (comma separated) for normalization\n')
        iss_list = string.split(',')
        n= -1                                                                       #tracks number of plots made
        for Listbox_num, Listbox_reference in enumerate(Listbox_list):
            for data_num in range(Listbox_reference.size()):
                if data_num in Listbox_reference.curselection():
                    filename = Listbox_reference.get(data_num)
                    full_filename = full_filenames_list[filenames.index(filename)]
                    n+=1
                    data_array = parse_file(filename,full_filename)
                    try:
                        iss = float(iss_list[n])
                    except:
                        try:
                            iss = float(iss_list[0])
                        except:
                            iss = 1e-9
                    try:
                        X,Y,Z = data_array
                    except:

                        x = data_array[:,0]
                        y = data_array[:,1]
                        z = data_array[:,2]
                        x0 = x[0]
                        y0 = y[0]

                        for i in range(1,x.size): # determine size of image
                            if (x[i] == x0) and (x[i] != x[i-1]): # find index where x repeats, if it is not slow scan direction
                                xpts = i
                                ypts = int(x.size / xpts)
                                break
                            if (y[i] == y0) and (y[i] != y[i-1]):
                                ypts = i
                                xpts = int(x.size / ypts)
                                break
                        X = x.reshape((ypts,xpts))
                        Y = y.reshape((ypts,xpts))
                        Z = z.reshape((ypts,xpts))
                        data[filename] = (X,Y,Z)



                    fig,ax = plt.subplots()
                    extent=[X[0,0], X[0,X.shape[1]-1], Y[0,0], Y[Y.shape[0]-1, 0]]
                    secm_window = SecmWindow(root,fig, Z, extent=extent, filename=filename, iss=iss, X=X, Y=Y)

                    fig_list.append(fig)
                    axes_list.append(ax)
                    SECM_plot_ref_list.append((fig,ax,secm_window.image,secm_window.colorbar))


    def parse_file(filename,full_filename):
        """Parse the data file for numeric data and append data array to the dictionary."""
        try: # if the file has been loaded previously, get it from the data dictionary
            data_array = data[filename]
            return data_array
        except KeyError: pass

        try: # if the file is a binary file saved by numpy
            data_array = np.load(full_filename, allow_pickle=True)
            return data_array
        except: pass

        try:
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
                The footer is likely absent or small, errors are less likely than
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

            reading_data = False
            for i,row in enumerate(text_file):
                if row[0] == '#': continue
                if not reading_data: metadata_row = i
                columns_list = row.strip().split(delim)
                for j,element in enumerate(columns_list):
                    if len(element) == 0: continue
                    try:                                            #if the row contains the correct number of elements and these are
                        numeric_element = float(element)            #all numeric type, it is the data we want to plot and store.
                        reading_data = True
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
            if filename not in metadata:
                metedata_text = ''
                for row in text_file[:metadata_row]: metedata_text += row
                metadata[filename] = metedata_text

            return data_array
        except:
            print('-'*20)
            print('Failed to load file')
            print('-'*20)


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

    def add_text(ax, event=None):

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
            file = filedialog.asksaveasfile(filetypes = [('Multi line graph', '*.mlg'), ('All Files', '*.*')], defaultextension = [('Multi line graph', '*.mlg'), ('All Files', '*.*')])
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

    def clear_axes(ax):
        ax.clear()
        ax.get_figure().canvas.draw()
        ax.get_figure().canvas.flush_events()

#%%--------------------------------secm plot popup menu commands---------------



#%%--------------------------------main window popup menu commands-------------




#%%

    def code_input():
        EDITOR_WIDTH = 10
        class CodeWindow(tk.Toplevel):
            def __init__(self, parent, *args, **kwargs):
                tk.Toplevel.__init__(self, parent, *args, **kwargs)

                self.OUTPUT_CLOSED = True           #track if output window is closed

                self.bind('<Control-Return>', lambda event: self.run(shortcut=True))
                self.auto_refresh = tk.IntVar()

                self.buttons_frame = tk.Frame(self)
                self.buttons_frame.grid(row=0, column=0, sticky='NS')

                self.spacer = tk.Label(self.buttons_frame, text='   ', width=EDITOR_WIDTH)
                self.spacer.grid(row=0, column=0, padx=10, pady=MAIN_PAD)

                self.btn_run = tk.Button(self.buttons_frame, text='Run', width=EDITOR_WIDTH, command=self.run)
                self.btn_run.grid(row=1, column=0, padx=10, pady=MAIN_PAD)

                self.btn_File = tk.Button(self.buttons_frame, text='Save', width=EDITOR_WIDTH, command=self.save)
                self.btn_File.grid(row=2, column=0, padx=10, pady=MAIN_PAD)

                self.chdirButton = tk.Button(self.buttons_frame, text='ChDir', width=EDITOR_WIDTH, command=self.change_dir)
                self.chdirButton.grid(row=3, column=0, padx=10, pady=MAIN_PAD)

                self.btn_Close = tk.Button(self.buttons_frame, text='Close', width=EDITOR_WIDTH, command=self.close)
                self.btn_Close.grid(row=4, column=0, padx=10, pady=MAIN_PAD)

                self.srefreshButton = tk.Checkbutton(self.buttons_frame, text = 'Refresh', variable = self.auto_refresh)
                self.srefreshButton.select()
                self.srefreshButton.grid(row=5,column = 0, padx=10, pady=MAIN_PAD)

                self.loc_Label = tk.Label(self.buttons_frame)
                self.loc_Label.grid(row=6,column = 0, padx=10, pady=MAIN_PAD)

                # Parent widget for the text editor
                self.text_Frame = tk.LabelFrame(self, text="Code", padx=5, pady=5)
                self.text_Frame.grid(row=0, column=1, rowspan=5, padx=MAIN_PAD, pady=MAIN_PAD, sticky='NSEW')

                self.columnconfigure(1, weight=1)
                self.rowconfigure(0, weight=1)

                self.text_Frame.rowconfigure(0, weight=1)
                self.text_Frame.columnconfigure(0, weight=1)

                # Create the textbox
                self.text_editor = scrolledtext.ScrolledText(self.text_Frame)
                self.text_editor.insert('1.0',
                                   '#from mihailpkg import cv_processing as cp\n' +
                                   '#ax = axes_list[0]\n')
                self.text_editor.bind('<KeyRelease>', self.show_loc)
                self.text_editor.bind('<ButtonRelease-1>', self.show_loc)

                self.text_editor.grid(row=0, column=0,   sticky='NSEW')

                self.protocol("WM_DELETE_WINDOW", self.close)

            def save(self):
                text = self.text_editor.get(1.0,tk.END)
                files = [('Python Files', '*.py'),
                         ('Text Document', '*.txt'),
                         ('All Files', '*.*')]
                file = filedialog.asksaveasfile(filetypes = files, defaultextension = files)
                file.write(text)
                file.close()

            def change_dir(self):
                directory = filedialog.askdirectory()
                os.chdir(directory)

            def show_loc(self, event=None):
                (line, char)= self.text_editor.index(tk.INSERT).split(".")
                self.loc_Label.config(text='r {} c {}'.format(line, char))

            def close(self):
                if self.OUTPUT_CLOSED == False:
                    self.output_window.on_closing()
                self.destroy()

            def run(self, shortcut=False):
                i = float(self.text_editor.index(tk.INSERT))
                if shortcut: self.text_editor.delete("insert-1c")
                text = self.text_editor.get(1.0, tk.END)
                namespace = {'data':data, 'fig_list':fig_list, 'axes_list':axes_list, 'plt':plt, 'np':np, 'os':os,
                             'Image':Image, 'ImageWindow':ImageWindow, 'root':root}
                if self.OUTPUT_CLOSED: self.make_output_window()
                exec(text, namespace)
                if self.auto_refresh.get():
                    for fig in fig_list:
                        fig.canvas.draw()
                        fig.canvas.flush_events()

            def make_output_window(self):
                self.output_window = OutputWindow(self)

        class OutputWindow(tk.Toplevel):
            def __init__(self, parent, *args, **kwargs):
                tk.Toplevel.__init__(self, parent, *args, **kwargs)
                self.parent = parent
                self.title('Output')
                self.text_output = scrolledtext.ScrolledText(self)
                self.text_output.tag_config('warning', background="yellow", foreground="red")
                self.text_output.pack(fill='both', expand=True)
                self.protocol("WM_DELETE_WINDOW", self.on_closing)
                self.parent.OUTPUT_CLOSED = False

                redirecter = StdoutRedirector(self.text_output)
                sys.stdout = redirecter
                sys.stderr = redirecter

            def on_closing(self):
                self.parent.OUTPUT_CLOSED = True
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                self.destroy()

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
                    label = 'text size',
                    command = self.text_size)
                self.Insert_menu.add_command(
                    label = 'SECM clim',
                    command = self.SECM_clim)
                self.Insert_menu.add_command(
                    label = 'edit SECM',
                    command = self.edit_SECM)
                self.Insert_menu.add_command(
                    label = 'norm. PACs',
                    command = self.norm_PACs)
                self.Insert_menu.add_command(
                    label = 'open image',
                    command = self.open_image)
                self.Insert_menu.add_command(
                    label = 'bgrd-sub pxrd',
                    command = self.bgrd_sub_pxrds)
                self.Insert_menu.add_command(
                    label = 'norm pxrd',
                    command = self.normalize_pxrds)

            def text_size(self):
                code_win.text_editor.insert(tk.END,
                                            "font_size = 14\n" +
                                            "ax.xaxis.get_label().set_fontsize(font_size)\n" +
                                            "ax.yaxis.get_label().set_fontsize(font_size)\n" +
                                            "ax.tick_params(axis='both', which='major', labelsize=font_size-1)\n" +
                                            "ax.legend(fontsize=font_size-1)")
            def SECM_clim(self):
                code_win.text_editor.insert(tk.END,
                                            "low,high = 0.5,1.5\n" +
                                            "image = ax.get_children()[0]\n" +
                                            "image.set_clim(low,high)")
            def edit_SECM(self):
                code_win.text_editor.insert(tk.END,
                                            "filename = \n" +
                                            "x,y,z = data[filename]\n" +
                                            "\n" +
                                            "data[filename] = x,y,z\n")

            def norm_PACs(self):
                code_win.text_editor.insert(tk.END,
                                            "a = 5"
                                            "lim_currents = []\n" +
                                            "x_offsets = []\n" +
                                            "for line, i_lim, x_off in zip(ax.lines, lim_currents, x_offsets):\n" +
                                            "\tline.set_xdata((line.get_xdata().max() - line.get_xdata() + x_off) / a)\n" +
                                            "\tline.set_ydata(line.get_ydata() / i_lim)\n" +
                                            "ax.relim()\n" +
                                            "ax.autoscale_view()")

            def open_image(self):
                code_win.text_editor.insert(tk.END,
                                            "filename = \n" +
                                            "ext = '.tif'\n" +
                                            "filename += ext\n" +
                                            "image = Image.open(filename)\n" +
                                            "img_window = ImageWindow(root,image, filename=filename[:-4])\n"
                                            "image_arr = np.array(image)\n" +
                                            "image_fft = np.fft.fft2(image_arr)\n"
                                            "img_window.fft = image_fft\n" +
                                            "image_fft_real = np.real(np.fft.fftshift(image_fft))\n" +
                                            "image = Image.fromarray(image_fft_real)\n" +
                                            "img_window = ImageWindow(root,image, filename=filename[:-4] + ' fft')\n")

            def bgrd_sub_pxrds(self):
                code_win.text_editor.insert(tk.END,
                                            "from mihailpkg import cv_processing as cp\n" +
                                            "def odd(x):\n" +
                                            	"\treturn round(x/2) * 2 - 1\n" +
                                            "ax = axes_list[1]\n" +
                                            "ax.clear()\n" +
                                            "\n" +
                                            "lam = 1e8\n" +
                                            "p = 0.01\n" +
                                            "\n" +
                                            "data = axes_list[0].lines\n" +
                                            "for i,line in enumerate(data):\n" +
                                            	"\tlabel = line.get_label()\n" +
                                            	"\tx,y = line.get_data()\n" +
                                            	"\tx,y = x.copy(), y.copy()\n" +
                                            	"\tbkrd = cp.baseline_als(y, lam=lam, p=p)\n" +
                                            	"\ty -= bkrd\n" +
                                            	"\ty = cp.savitzky_golay(y,odd(y.size/200),10)\n" +
                                            	"\tax.plot(x, y, label=label)\n" +
                                            "ax.get_yaxis().set_visible(False)\n" +
                                            "#ax.text(0.5, 0.5, 'lam: {:.2e} p: {:.2f}'.format(lam,p), transform=ax.transAxes)\n" +
                                            "ax.legend()\n")

            def normalize_pxrds(self):
                code_win.text_editor.insert(tk.END,
                                            "from mihailpkg import cv_processing as cp\n" +
                                            "def odd(x):\n" +
                                            	"\treturn round(x/2) * 2 - 1\n" +
                                            "ax = axes_list[2]\n" +
                                            "ax.clear()\n" +
                                            "\n" +
                                            "lam = 1e8\n" +
                                            "p = 0.01\n" +
                                            "\n" +
                                            "data = axes_list[1].lines\n" +
                                            "for i,line in enumerate(data):\n" +
                                            	"\tlabel = line.get_label()\n" +
                                            	"\tx,y = line.get_data()\n" +
                                            	"\tx,y = x.copy(), y.copy()\n" +
                                            	"\ty /= y.max()\n" +
                                            	"\ty -= i\n" +
                                            	"\tax.plot(x, y, label=label)\n" +
                                            "ax.get_yaxis().set_visible(False)\n" +
                                            "#ax.text(0.5, 0.5, 'lam: {:.2e} p: {:.2f}'.format(lam,p), transform=ax.transAxes)\n" +
                                            "ax.legend()\n")



            def save_data(self):
                import pickle
                file = filedialog.asksaveasfile(mode = 'wb', filetypes = [('Pickled Files', '*.pickle')], defaultextension = [('Pickled Files', '*.pickle')])
                if file: pickle.dump(data,file)

        class IORedirector():
            def __init__(self,text_area):
                self.text_area = text_area

        class StdoutRedirector(IORedirector):
            '''A class for redirecting stdout to this Text widget.'''
            def write(self,output):
                if 'Exception' in output or 'Error' in output:
                    self.text_area.insert('end',output, 'warning')
                else:
                    self.text_area.insert('end',output)
            def flush(self):
                pass

        code_win = CodeWindow(root)
        menubar = EditorAppMenubar(code_win)

    def replot_secm(fig):
        pass

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
            self.File_menu.add_command(
                label = 'open lineplot',
                command = self.open_lineplot)


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
                    new_filename = simpledialog.askstring(title ='',
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
                            filenames[m] = new_filename

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

        def open_lineplot(self):
            n = simpledialog.askinteger('Open lineplot', 'fig number')
            fig = fig_list[n]
            line_plot_window = LinePlotWindow(root,fig)
            line_plot_window.title('fig ' + str(fig.number - 1))
            line_plot_window.refresh()

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
            add_desc('Up', 'Upscale image')
            add_desc('Down', 'Downscale image')
            add_desc('Right', 'Next image')
            add_desc('Left', 'Previous image')
            add_desc('m', 'Measure image')
            add_desc('t', 'Add text')


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

            self.image_Button = tk.Button(self,text='Plot Image', command = lambda: plot_curve(image_file=True))
            self.image_Button.grid(row = 4, column = 0, columnspan = 1, sticky='nesw', padx=MAIN_PAD, pady=MAIN_PAD)

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
                    command = self.parent.trim_secm)
                self.add_command(
                    label = 'save bitmap',
                    command = self.parent.save_bitmap)
                self.add_command(
                    label = 'save data',
                    command = self.parent.save_data)


        def __init__(self, parent, fig, Z, extent=None, filename='', iss=1, X=None, Y=None, *args, **kwargs):
            tk.Toplevel.__init__(self, parent, *args, **kwargs)

            self.fig = fig
            self.fig.set_size_inches(6,5)
            self.ax = fig.axes[0]

            self.X = X # assumed to be in μm
            self.Y = Y # assumed to be in μm
            self.Z = Z # assumed to be in A
            self.iss = iss
            self.Z_norm = self.Z / self.iss

            self.extent  = extent
            self.filename = filename


            self.image = self.ax.imshow(self.Z_norm, extent=self.extent, origin='lower', interpolation='bicubic', cmap='Spectral_r')
            self.image.original_vmin, self.image.original_vmax = self.image.get_clim()
            self.image.vmin, self.image.vmax = self.image.get_clim()

            self.colorbar = self.fig.colorbar(self.image, ax=self.ax, format = '%.2f')
            self.fig.set_size_inches(6,5)

            self.title('fig ' + str(self.fig.number - 1) + ' | ' + self.filename + ' | ' + str(self.iss))

            self.ax.set_xlabel('Distance / $\mu$m')
            self.ax.set_ylabel('Distance / $\mu$m')
            self.ax.set_aspect('equal')

            self.app_rclick_menu = SecmWindow.AppRclickMenu(self,self.ax)

            self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="bottom", fill='both',expand=True)
            self.canvas.get_tk_widget().bind('<Button-3>', lambda x: self.popup_menu(event=x))
            self.canvas.get_tk_widget().bind('t', lambda x: add_text(self.ax, event=x))
            self.canvas.get_tk_widget().bind('r', self.refresh)

            self.toolbarFrame = tk.Frame(master=self)
            self.toolbarFrame.pack(side="top",fill='x',expand=False)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)

        def popup_menu(self, event = None):
            try:
                self.app_rclick_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.app_rclick_menu.grab_release()

        def refresh(self, event=None):
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def trim_secm(self):
            def slider_changed(event):
                self.image.vmin = lower_index_slider.get()
                self.image.vmax = upper_index_slider.get()
                self.image.set_clim([self.image.vmin, self.image.vmax])
                self.refresh()

            win = tk.Toplevel(self)

            resolution = (self.image.original_vmax - self.image.original_vmin) / 50

            lower_index_slider = tk.Scale(win, from_= self.image.original_vmin, to = self.image.original_vmax, orient=tk.HORIZONTAL, length=600, command=slider_changed, resolution = resolution)
            lower_index_slider.set(self.image.original_vmin)
            lower_index_slider.pack()

            upper_index_slider = tk.Scale(win, from_= self.image.original_vmin, to=self.image.original_vmax, orient=tk.HORIZONTAL, length=600, command=slider_changed, resolution = resolution)
            upper_index_slider.set(self.image.original_vmax)
            upper_index_slider.pack()

            tk.Button(win, text='Close', command=win.destroy).pack()

        def save_bitmap(self):
            filename = filedialog.asksaveasfilename(filetypes = [('Tiff', '*.tif'), ('All Files', '*.*')], defaultextension=[('Tiff', '*.tif'), ('All Files', '*.*')])
            arr = self.Z_norm.clip(self.image.vmin, self.image.vmax)[::-1,:] # trim data and set origin as lower left
            arr = (arr - self.image.vmin)/(self.image.vmax - self.image.vmin) # normalize 0 to 1
            img = Image.fromarray(np.uint8(cm.Spectral_r(arr)*255))
            img.save(filename)

        def save_data(self):

            if type(self.X) != np.ndarray or type(self.Y) != np.ndarray:
                return
            elif self.X.shape != self.Z.shape or self.Y.shape != self.Z.shape:
                print('Error: X and Y must be same shape as Z')
                return
            else:
                arr = np.zeros((self.Z.size,3))
                arr[:,0] = self.X.reshape(self.Z.size)
                arr[:,1] = self.Y.reshape(self.Z.size)
                arr[:,2] = self.Z.reshape(self.Z.size)
                filename = filedialog.asksaveasfilename(filetypes = [('Text', '*.txt'), ('All Files', '*.*')], defaultextension=[('Text', '*.txt'), ('All Files', '*.*')])
                header = 'X / μm\tY / μm\tZ / A'
                np.savetxt(filename, arr, delimiter='\t', fmt='%.3e')

#%%--------------------------------build the line plot window------------------

    class ImageWindow(tk.Toplevel):
        class AppRclickMenu(tk.Menu):
            def __init__(self, parent, *args, **kwargs):
                tk.Menu.__init__(self, parent, *args, **kwargs)
                self.parent = parent

                self.add_command(
                    label = 'measure',
                    command = self.parent.measure)

                self.add_command(
                    label = 'add text',
                    command = self.parent.text)

                self.add_command(
                    label = 'append instance',
                    command = self.parent.append_instance)
                self.add_command(
                    label = 'calc fft',
                    command = self.parent.calc_fft)
                self.add_command(
                    label = 'histogram',
                    command = self.parent.histogram)
                self.add_command(
                    label = 'Auto B/C',
                    command = self.parent.auto_BC)
                self.add_command(
                    label = 'close',
                    command = self.parent.destroy)

        def __init__(self, parent, image, filename='', *args, **kwargs):
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
            self.filename = filename
            self.image = image
            self.original_image = self.image.copy()
            self.tkimage = ImageTk.PhotoImage(self.original_image)

            #self.resizable(False,False)
            self.geometry("{}x{}".format(self.tkimage.width(), self.tkimage.height()))
            self.title("{}x{}\t".format(self.tkimage.width(), self.tkimage.height()) + self.filename)

            self.canvas = tk.Canvas(self, width=self.tkimage.width(), height=self.tkimage.height())
            self.image_container = self.canvas.create_image(0, 0, anchor='nw', image=self.tkimage)
            self.canvas.pack(anchor='nw')

            self.magnification = 1

            self.bind('<Button-3>', lambda x: self.popup_menu(event=x))
            self.bind('<Right>', lambda x: self.open_next_previous(event=x))
            self.bind('<Left>', lambda x: self.open_next_previous(event=x, previous=True))
            self.bind('<MouseWheel>', lambda x: self.scale_mousewheel(event=x, magnification=(4/3)))
            self.bind('<Up>', lambda x: self.scale(event=x, magnification=(4/3)))
            self.bind('<Down>', lambda x: self.scale(event=x, magnification=(3/4)))
            self.bind('<m>', lambda x: self.measure(event=x))
            self.bind('<Control-s>', lambda x: self.save(event=x))
            self.bind('<t>', lambda x: self.text(event=x))
            self.bind('<Double-Button-1>', lambda x: self.return_to_origin(event=x))
            self.bind('<ButtonPress-2>', self.move_from)
            self.bind('<B2-Motion>', self.move_to)
            self.bind('<h>', self.histogram)
            self.bind('<f>', self.calc_fft)

            self.app_rclick_menu = ImageWindow.AppRclickMenu(self)

        def popup_menu(self, event = None):
            try:
                self.app_rclick_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.app_rclick_menu.grab_release()

        def move_from(self, event):
            ''' Remember previous coordinates for scrolling with the mouse '''
            self.x, self.y = event.x, event.y

        def move_to(self, event):
            ''' Drag (move) canvas to the new position '''
            dx = event.x - self.x
            dy = event.y - self.y
            self.canvas.move(self.image_container, dx, dy)
            self.x, self.y = event.x, event.y


        def return_to_origin(self, event):
            self.geometry("{}x{}".format(self.tkimage.width(), self.tkimage.height()))
            x,y = self.canvas.coords(self.image_container)
            self.canvas.move(self.image_container, -x, -y)

        def update_image(self):
            self.tkimage = ImageTk.PhotoImage(self.image)
            #self.geometry("{}x{}".format(self.tkimage.width(), self.tkimage.height()))
            self.canvas.itemconfig(self.image_container, image=self.tkimage)
            self.canvas.config(width=self.tkimage.width(), height=self.tkimage.height())
            self.geometry("{}x{}".format(self.tkimage.width(), self.tkimage.height()))

        def append_instance(self, event=None):
            data[self.filename] = self

        def save(self, event=None):
            filename = filedialog.asksaveasfilename(filetypes = [('Tiff', '*.tif'), ('All Files', '*.*')], defaultextension=[('Tiff', '*.tif'), ('All Files', '*.*')])
            self.image.save(filename)

        def open_next_previous(self, event = None, previous = False):
            #ensure only a single file is selected
            number_of_selected_files = 0

            for Listbox_ in Listbox_list:
                for i in Listbox_.curselection():
                    number_of_selected_files += 1
                    Listbox = Listbox_
            if number_of_selected_files > 1:
                print("{} files selected.\nPlease select only one, dumb bitch".format(number_of_selected_files))
                return

            #move selection up by 1
            Listbox.selection_clear(i)
            if previous == False:
                i += 1
                if i == Listbox.size(): i = 0
                filename = Listbox.get(i)
                Listbox.selection_set(i)
            else:
                Listbox.selection_clear(i)
                i -= 1
                if i < 0: i = Listbox.size() - 1
                filename = Listbox.get(i)
                Listbox.selection_set(i)

            self.original_image = Image.open(full_filenames_list[filenames.index(filename)])
            self.image = self.original_image.copy()
            self.filename = filename

            if not (0.99 < self.magnification < 1.01):
                w, h = self.original_image.width, self.original_image.height
                self.image = self.original_image.resize((round(w*self.magnification), round(h*self.magnification)))
                self.title("{}x{} {:.0f}%\t".format(self.image.width, self.image.height, self.magnification*100) + self.filename)
            else:
                self.title("{}x{}\t".format(self.image.width, self.image.height) + self.filename)

            self.update_image()
            data[filename] = self.original_image

        def scale(self, event = None, magnification = 1):

            self.magnification *= magnification
            w, h = self.original_image.width, self.original_image.height

            if 0.99 < self.magnification < 1.01:
                self.image = self.original_image.copy()
                self.title("{}x{}\t".format(self.image.width, self.image.height) + self.filename)

            else:
                self.image = self.original_image.resize((round(w*self.magnification), round(h*self.magnification)))
                self.title("{}x{} {:.0f}%\t".format(self.image.width, self.image.height, self.magnification*100) + self.filename)

            self.tkimage = ImageTk.PhotoImage(self.image)
            self.update_image()

        def scale_mousewheel(self, event = None, magnification = 1):
            if event.delta > 1:
                self.scale(magnification = 4/3)
            if event.delta < 1:
                self.scale(magnification = 3/4)


        def calc_fft(self, event=None):

            image_arr = np.float32(self.image) # uint to float

            if len(image_arr.shape) == 3: # if image format is RGB or RGBA
                image_arr = 0.2989*image_arr[:,:,0] + 0.5870*image_arr[:,:,1] + 0.1140*image_arr[:,:,2]

            image_fft = np.fft.fft2(image_arr)
            image_fft_real = np.log(np.abs(np.fft.fftshift(image_fft)))
            i_min,i_max = np.min(image_fft_real), np.max(image_fft_real)
            image_fft_real = (image_fft_real - i_min) / (i_max - i_min) * 255 # equalize histogram and convert to uint8
            image_fft_real = np.uint8(image_fft_real)

            size = max(image_arr.shape)
            image = Image.fromarray(image_fft_real).resize((size,size))

            img_window = ImageWindow(root,image, filename=self.filename + ' fft')
            img_window.fft = image_fft

        def histogram(self, event=None):
            image_arr = np.float32(self.image) # uint to float

            if len(image_arr.shape) == 3: # if image format is RGB or RGBA
                fig, ax = plt.subplots(ncols=3, figsize=(9,3))
                line_plot_window = LinePlotWindow(root,fig)
                line_plot_window.title(self.filename)
                colors = ['r','g','b']

                for i in range(3):
                    nrow,ncol = image_arr[:,:,i].shape
                    arr = image_arr[:,:,i].reshape(nrow*ncol)
                    ax[i].hist(arr, bins=np.linspace(0,255,256), color=colors[i])
                    ax[i].axes.get_yaxis().set_visible(False)
                    ax[i].spines['right'].set_visible(False);ax[i].spines['left'].set_visible(False);ax[i].spines['top'].set_visible(False)
                    ax[i].tick_params(axis='both', which='major', labelsize=8)
                    ax[i].set_xlim(0,255)
                    fig.canvas.draw()
                    fig.canvas.flush_events()


            else:
                nrow,ncol = image_arr.shape
                arr = image_arr.reshape(nrow*ncol)
                fig,ax = plt.subplots(figsize=(3,3))
                line_plot_window = LinePlotWindow(root,fig)
                line_plot_window.title(self.filename)
                ax.hist(arr, bins=np.linspace(0,255,256))
                ax.axes.get_yaxis().set_visible(False)
                ax.spines['right'].set_visible(False);ax.spines['left'].set_visible(False);ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.set_xlim(0,255)
                fig.canvas.draw()
                fig.canvas.flush_events()

        def auto_BC(self, event=None):
            image = self.image.convert('RGB')
            image = ImageOps.equalize(image)
            img_window = ImageWindow(root,image, filename=self.filename + ' equalized')


        def measure(self, event=None):
            WIDTH = 10
            def return_pointer_coord(event=None):
                x0, y0 = self.canvas.coords(self.image_container)
                x, y = event.x - x0, event.y - y0
                if not pt1.active:
                    pt1.active = True
                    pt1.config(text='{}, {}'.format(x, y))
                    pt2.config(text='')
                    dist.config(text='')
                    pt1.x, pt1.y = x, y
                    return

                if pt1.active:
                    pt1.active = False
                    pt2.config(text='{}, {}'.format(x, y))
                    pt2.x, pt2.y = x, y
                    distance = ((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5
                    dist.config(text='{:.0f} px'.format(distance),fg='red')
                    xy = [(pt1.x, pt1.y),(pt2.x, pt2.y)]
                    try:
                        self.previous_image = self.image.copy()
                        draw = ImageDraw.Draw(self.image)
                        draw.line(xy, fill='red')
                    except:
                        self.image = self.image.convert(mode='RGBA')
                        self.previous_image = self.image.copy()
                        draw = ImageDraw.Draw(self.image)
                        draw.line(xy, fill='red')
                    self.update_image()

                    return

            def close_window():
                self.canvas.unbind('<Button-1>')
                self.bind('<Right>', lambda x: self.open_next_previous(event=x))
                self.bind('<Left>', lambda x: self.open_next_previous(event=x, previous=True))
                self.bind('<Up>', lambda x: self.scale(event=x, magnification=(4/3)))
                self.bind('<Down>', lambda x: self.scale(event=x, magnification=(3/4)))
                self.bind('<m>', lambda x: self.measure(event=x))
                self.bind('<MouseWheel>', lambda x: self.scale_mousewheel(event=x, magnification=(4/3)))
                self.image = self.original_image.copy()
                self.update_image()


                win.destroy()

            def measure(event=None):
                if measure_button.cget('text') == 'Measure' or measure_button.cget('text') == 'Measure Off':
                    self.canvas.bind('<Button-1>', lambda x: return_pointer_coord(event=x))
                    measure_button.config(text='Measure On')
                    return

                if measure_button.cget('text') == 'Measure On':
                    self.canvas.unbind('<Button-1>')
                    measure_button.config(text='Measure Off')
                    return
            def undo(event=None):
                self.image = self.previous_image
                self.update_image()

            win = tk.Toplevel(root)
            win.protocol('WM_DELETE_WINDOW', close_window)
            win.resizable(False,False)

            self.bind('<Control-z>', undo)
            self.unbind('<Right>')
            self.unbind('<Right>')
            self.unbind('<Left>')
            self.unbind('<Up>')
            self.unbind('<Down>')
            self.unbind('<m>')
            self.unbind('<MouseWheel>')

            tk.Label(win, text='pt1:', width=WIDTH).grid(row=0, column=0)
            pt1 = tk.Label(win, text='', width=WIDTH)
            pt1.active = False
            pt1.grid(row=0, column=1)
            tk.Label(win, text='pt2:', width=WIDTH).grid(row=1, column=0)
            pt2 = tk.Label(win, text='', width=WIDTH)
            pt2.grid(row=1, column=1)
            tk.Label(win, text='dist:', width=WIDTH).grid(row=2, column=0)
            dist = tk.Label(win, text='', width=WIDTH)
            dist.grid(row=2, column=1)

            measure_button = tk.Button(win, text='Measure', width=10, command=measure)

            measure_button.grid(row=3, column=0, columnspan=1)
            close_button = tk.Button(win, text='Close', command=close_window, width=WIDTH)
            close_button.grid(row=3, column=1, columnspan=1)

        def text(self, event=None):
            WIDTH = 20
            def return_pointer_coord(event=None):
                x0, y0 = self.canvas.coords(self.image_container)
                x, y = event.x - x0, event.y - y0
                try:
                    self.previous_image = self.image.copy()
                    draw = ImageDraw.Draw(self.image)
                    draw.text((x,y), textEntry.get(), fill='red')
                except:
                    self.image = self.image.convert(mode='RGBA')
                    self.previous_image = self.image.copy()
                    draw = ImageDraw.Draw(self.image)
                    draw.text((x,y), textEntry.get(), fill='red')

                self.update_image()

            def close_window():
                self.canvas.unbind('<Button-1>')
                self.bind('<Right>', lambda x: self.open_next_previous(event=x))
                self.bind('<Left>', lambda x: self.open_next_previous(event=x, previous=True))
                self.bind('<Up>', lambda x: self.scale(event=x, magnification=(4/3)))
                self.bind('<Down>', lambda x: self.scale(event=x, magnification=(3/4)))
                self.bind('<t>', lambda x: self.text(event=x))
                self.image = self.original_image.copy()
                self.update_image()
                win.destroy()

            def text(event=None):
                if text_button.cget('text') == 'Text' or text_button.cget('text') == 'Text Off':
                    self.canvas.bind('<Button-1>', lambda x: return_pointer_coord(event=x))
                    text_button.config(text='Text On')
                    return

                if text_button.cget('text') == 'Text On':
                    self.canvas.unbind('<Button-1>')
                    text_button.config(text='Text Off')
                    return

            def undo(event=None):
                self.image = self.previous_image
                self.update_image()

            win = tk.Toplevel(root)
            win.protocol('WM_DELETE_WINDOW', close_window)
            win.resizable(False,False)

            self.bind('<Control-z>', undo)
            self.unbind('<Right>')
            self.unbind('<Left>')
            self.unbind('<Up>')
            self.unbind('<Down>')
            self.unbind('<t>')

            tk.Label(win, text='Text to insert', width=WIDTH).grid(row=0, column=0, columnspan=2)
            textEntry = tk.Entry(win, text='', width=WIDTH)
            textEntry.grid(row=1, column=0, columnspan=2, pady=10)

            text_button = tk.Button(win, text='Text', width=10, command=text)
            text_button.grid(row=3, column=0, columnspan=1)
            close_button = tk.Button(win, text='Close', command=close_window, width=10)
            close_button.grid(row=3, column=1, columnspan=1)

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
                self.add_command(
                    label = 'clear plot',
                    command = lambda: clear_axes(ax))


        def __init__(self, parent, fig, *args, **kwargs):
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
            self.fig = fig
            self.ax = fig.axes[0]

            self.app_rclick_menu = LinePlotWindow.AppRclickMenu(self,self.ax)


            self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="bottom",fill='both',expand=True)
            self.canvas.get_tk_widget().bind('<Double-Button-1>', self.popup_menu)
            self.canvas.get_tk_widget().bind('l', lambda x: legend_ON(ax = self.ax, event=x))
            self.canvas.get_tk_widget().bind('<Shift-L>', lambda x: legend_OFF(ax = self.ax, event=x))
            self.canvas.get_tk_widget().bind('r', self.refresh)
            self.canvas.get_tk_widget().bind('t', lambda x: add_text(self.ax, event=x))


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
            filenameslist = filedialog.askopenfilenames(filetypes = [('All files','*.*'),
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

if __name__ == '__main__':
    main()
