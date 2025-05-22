import tkinter as tk
from tkinter import ttk
from capture import capture_data
from train import train_model
from recognize import recognize_faces

def main():
    win = tk.Tk()
    win.title("Face Recognition with FaceNet")
    win.geometry('400x300')
    win.configure(bg='#263D42')

    ttk.Label(win, text="Face Recognition System (FaceNet)", background="grey",
            foreground="white", font=("Arial", 14)).place(x=50, y=20)
    ttk.Label(win, text="ID:", background="#263D42",
              foreground="white").place(x=50, y=80)
    ttk.Label(win, text="Name:", background="#263D42",
              foreground="white").place(x=50, y=120)

    int1 = tk.StringVar()
    str1 = tk.StringVar()
    edit_id = ttk.Entry(win, textvariable=int1,width=40)
    edit_id.place(x=100, y=80)
    edit_id.focus()
    edit_name = ttk.Entry(win, textvariable=str1, width=40)
    edit_name.place(x=100, y=120)

    ttk.Button(win, text="Capture Data", command=lambda: capture_data(edit_id.get(),
                edit_name.get())).place(x=50, y=180)
    ttk.Button(win, text="Train", command=train_model).place(x=170, y=180)
    ttk.Button(win, text="Recognize", command=recognize_faces).place(x=290, y=180)

    win.mainloop()

if __name__ == "__main__":
    main()
