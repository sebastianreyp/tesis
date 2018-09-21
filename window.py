from tkinter import *

root = Tk()
root.geometry('1200x600')
root.title("Registration Form")

title = Label(root, text="Análisis de sentimiento",width=20,font=("bold", 20))
title.place(x=0,y=0)

Button(root, text='Extraer',width=15,bg='red',fg='white',font=("bold", 10)).place(x=10,y=50)
Button(root, text='Limpiar',width=15,bg='blue',fg='white',font=("bold", 10)).place(x=150,y=50)
Button(root, text='Procesar',width=15,bg='green',fg='white',font=("bold", 10)).place(x=290,y=50)

entry_one = Text(root, width=40, height=30)
entry_one.place(x=10,y=100)

entry_two = Text(root, width=40, height=30)
entry_two.place(x=350,y=100)

root.mainloop()