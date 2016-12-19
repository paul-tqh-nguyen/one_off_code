
import os
import sys
import numpy
import Tkinter
import Image
import ImageTk

DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_X_POS = 200
DEFAULT_Y_POS = 50
DISPLAY_STRING_MAX_LENGTH = 160

def clamp_string(string, length):
    if len(string)>length:
        print "WARNING: String Clipped"
        print "             \""+string+"\""
        print "         has been clipped to "+string
        print "             \""+string[:length]+"\""
        return string[:length]
    else:
        return string

class Window(Tkinter.Frame):
      
    def __init__(self, title="Title", bg_color="white", width=DEFAULT_WINDOW_WIDTH, height=DEFAULT_WINDOW_HEIGHT, x_pos=DEFAULT_X_POS, y_pos=DEFAULT_Y_POS):
        self.parent = Tkinter.Tk()
        Tkinter.Frame.__init__(self, self.parent, background="white")
        self.set_title(title)
        self.pack(fill=Tkinter.BOTH, expand=1)
        self.width = width
        self.height = height
        self.set_dimensions(width=self.width, height=self.height, x_pos=x_pos, y_pos=y_pos)
        self.center()
        self.widget_list=[]
    
    def get_widget(self, widget_id):
        return self.widget_list[widget_id]
    
    def get_text_from_entry_box(self, widget_id):
        assert widget_id < len(self.widget_list), "widget_id ("+str(widget_id)+") out of bounds (we  have len(widget_list) == "+str(len(widget_list))+")"
        assert isinstance(self.widget_list[widget_id], Tkinter.Entry), "widget_list["+str(widget_id)+"] is not an instance of Tkinter.Entry"
        return  self.widget_list[widget_id].get()
    
    def update_image_label(self, widget_id, image_location=''):
        assert widget_id < len(self.widget_list), "widget_id ("+str(widget_id)+") out of bounds (we  have len(widget_list) == "+str(len(self.widget_list))+")"
        assert isinstance(self.widget_list[widget_id], Tkinter.Label), "widget_list["+str(widget_id)+"] is not an instance of Tkinter.Label"
        if image_location == '':
            self.get_widget(widget_id).configure(image = None)
            return
        photo = ImageTk.PhotoImage(Image.open(image_location))
        self.get_widget(widget_id).configure(image = photo)
        self.get_widget(widget_id).image = photo
        
    def add_image_label(self, image_location='', x_pos=10, y_pos=10):
        # returns ID, i.e. the position in widget_list
        widget_id = len(self.widget_list)
        if image_location=='':
            label = Tkinter.Label()
        else:
            photo = ImageTk.PhotoImage(Image.open(image_location))
            label = Tkinter.Label(self, image=photo, bg='white')
            label.image = photo
        label.place(x=x_pos, y=y_pos)
        self.widget_list.append( label )
        return widget_id
    
    def add_text_entry_box(self, default_text="", width=100, x_pos=10, y_pos=10):
        # returns ID, i.e. the position in widget_list
        widget_id = len(self.widget_list)
        entry = Tkinter.Entry(self, width=100)
        entry.place(x=x_pos, y=y_pos)
        entry.delete(0, Tkinter.END)
        entry.insert(0, default_text)
        self.widget_list.append(entry)
        return widget_id
    
    def add_text_label(self, text='', x_pos=10, y_pos=10):
        # returns ID, i.e. the position in widget_list
        # text label widgets will be stored as (Tkinter.Label, Tkinter.StringVar)
        widget_id = len(self.widget_list)
        string_var = Tkinter.StringVar()
        string_var.set(clamp_string(text, DISPLAY_STRING_MAX_LENGTH))
        label = Tkinter.Label(self, textvariable=string_var, bg='white')
        label.place(x=x_pos, y=y_pos)
        self.widget_list.append( (label, string_var) )
        return widget_id
    
    def update_text_label(self, widget_id, new_text=''):
        assert widget_id < len(self.widget_list), "widget_id ("+str(widget_id)+") out of bounds (we  have len(widget_list) == "+str(len(self.widget_list))+")"
        assert isinstance(self.widget_list[widget_id], tuple), "widget_list["+str(widget_id)+"] is not an instance of tuple"
        assert len(self.widget_list[widget_id]) == 2, "len(widget_list["+str(widget_id)+"]) != 2"
        assert isinstance(self.widget_list[widget_id][0], Tkinter.Label), "widget_list["+str(widget_id)+"][0] is not an instance of Tkinter.Label"
        assert isinstance(self.widget_list[widget_id][1], Tkinter.StringVar), "widget_list["+str(widget_id)+"][1] is not an instance of Tkinter.StringVar"
        self.get_widget(widget_id)[1].set(clamp_string(new_text, DISPLAY_STRING_MAX_LENGTH))
    
    def add_button(self, text="Button", command=lambda x: None, x_pos=10, y_pos=10):
        # returns ID, i.e. the position in widget_list
        widget_id = len(self.widget_list)
        button = Tkinter.Button(self, text=text, command=command)
        button.place(x=x_pos, y=y_pos)
        self.widget_list.append(button)
        return widget_id
    
    def start(self):
        self.parent.mainloop()
    
    def set_dimensions(self, width=DEFAULT_WINDOW_WIDTH, height=DEFAULT_WINDOW_HEIGHT, x_pos=DEFAULT_X_POS, y_pos=DEFAULT_Y_POS):
        self.width = width
        self.height = height
        self.parent.geometry( str(width)+'x'+str(height)+'+'+str(x_pos)+'+'+str(y_pos) )
    
    def set_title(self, new_title="Title"):
        self.parent.title(new_title)
    
    def center(self):
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        x_pos = (screen_width - self.width)/2
        y_pos = (screen_height - self.height)/2
        self.set_dimensions(width=self.width, height=self.height, x_pos=x_pos, y_pos=y_pos)
    
