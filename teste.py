import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("Dark")  
customtkinter.set_default_color_theme("dark-blue")  


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Diagnóstico de Esteatose Hepática em Exames de Ultrassom")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Esteatose Hepática", font=customtkinter.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.load_image_button = customtkinter.CTkButton(self.sidebar_frame, text="Carregar Imagem e Histograma", command=self.load_image,width=200)
        self.load_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Recortar ROI", command=self.sidebar_button_event,width=200)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Visualizar ROI e Histograma", command=self.sidebar_button_event,width=200)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Computar GLCM", command=self.sidebar_button_event,width=200)
        self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Caracterizar ROI", command=self.sidebar_button_event,width=200)
        self.sidebar_button_4.grid(row=5, column=0, padx=20, pady=10)
        
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Classificar Imagem", command=self.sidebar_button_event,width=200)
        self.sidebar_button_5.grid(row=6, column=0, padx=20, pady=10)

        


        # Create a label to display the image
        self.image_label = customtkinter.CTkLabel(self, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")


    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk  # Keep a reference to the image to prevent garbage collection

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def sidebar_button_event(self):
        print("sidebar_button click")


if __name__ == "__main__":
    app = App()
    app.mainloop()
