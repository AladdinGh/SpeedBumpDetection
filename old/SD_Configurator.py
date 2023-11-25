from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter as tk
window = Tk()
window.title("Scenario Detection Configurator")
window.geometry('600x600')

class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
chk_1 = BooleanVar()
chk_1.set(True) #set check state
chk_2 = BooleanVar()
chk_2.set(True) #set check state
chk_3 = BooleanVar()
chk_3.set(True) #set check state
chkKPI = Checkbutton(window, text='Case KPI :',var = chk_1)
chkStau = Checkbutton(window, text='Case Stau :',var = chk_2)
chkACASidabled = Checkbutton(window, text='Case ACA Disabled :',var = chk_3)

chkKPI.grid(column=0, row=0)
button1_ttp = CreateToolTip(chkKPI, \
 'Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, '
  'Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, ')
chkStau.grid(column=0, row=1)
chkACASidabled.grid(column=0, row=2)

#selected  disabled alternate
if (chkKPI.instate(['alternate'])):  
    print (chkKPI.state())
#messagebox.showinfo('Message title','Message content')



window.mainloop()