# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:20:50 2023

@author: Mihail
"""

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
