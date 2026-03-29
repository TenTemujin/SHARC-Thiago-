import sys
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.widgets import Meter

app = tb.Window(themename="cosmo")

# Simulate sidebar background
sidebar = tb.Frame(app, bootstyle="light", padding=20)
sidebar.pack(fill="y", expand=True)

# Try Meter
m = Meter(sidebar, bootstyle="primary", amountused=50)
m.pack()

# Try Button
btn = tb.Button(sidebar, bootstyle="secondary-link", text="Test Link Button")
btn.pack()

# To inspect the background of Meter's canvas and if it's white:
from PIL import ImageGrab
app.update_idletasks()
app.after(500, lambda: app.destroy())
app.mainloop()

print("Canvas bg:", m._canvas.cget('bg'))
print("Meter frame bg:", m.cget('style'))
