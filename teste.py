import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import scipy.io

customtkinter.set_appearance_mode("Dark")  
customtkinter.set_default_color_theme("dark-blue")  

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configuração da janela
        self.title("Diagnóstico de Esteatose Hepática em Exames de Ultrassom")
        self.geometry(f"{1100}x{580}")

        # Configuração do layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Criação da barra lateral com botões
        self.sidebar_frame = customtkinter.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Esteatose Hepática", font=customtkinter.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.load_image_button = customtkinter.CTkButton(self.sidebar_frame, text="Carregar Imagem e Histograma", command=self.load_image, width=200)
        self.load_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Recortar ROI", command=self.recortar_roi, width=200)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Visualizar ROI e Histograma", command=self.sidebar_button_event, width=200)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Computar GLCM", command=self.sidebar_button_event, width=200)
        self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Caracterizar ROI", command=self.sidebar_button_event, width=200)
        self.sidebar_button_4.grid(row=5, column=0, padx=20, pady=10)
        
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Classificar Imagem", command=self.sidebar_button_event, width=200)
        self.sidebar_button_5.grid(row=6, column=0, padx=20, pady=10)

        # Label para exibir a imagem
        self.image_label = customtkinter.CTkLabel(self, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Variáveis iniciais
        self.img = None
        self.img_resized = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk  # Manter referência da imagem

    def recortar_roi(self):
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            data_array = mat_data['data']

            # Obter o campo 'images'
            images = data_array['images']

            # Criar uma lista para armazenar todas as imagens
            image_list = []

            # Percorrer todos os pacientes e imagens
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]  # Acessar as imagens do paciente
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    image_list.append(img)

            print("Total de imagens carregadas:", len(image_list))

            # Selecionar a primeira imagem como exemplo
            image_data = image_list[0]

            # Verificar se image_data é um array NumPy
            if not isinstance(image_data, np.ndarray):
                tkinter.messagebox.showerror("Erro", "Os dados da imagem não são válidos.")
                return

            # Normalizar e converter a imagem
            if image_data.dtype != np.uint8:
                image_data = (255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))).astype(np.uint8)

            self.img = Image.fromarray(image_data)
            if self.img.mode != 'L':
                self.img = self.img.convert('L')

            # Criar janela para seleção da ROI
            self.crop_window = tkinter.Toplevel(self)
            self.crop_window.title("Selecione a ROI")
            self.crop_window.geometry("600x600")

            self.canvas = tkinter.Canvas(self.crop_window, width=600, height=600)
            self.canvas.pack()

            # Redimensionar a imagem para o canvas
            self.img_resized = self.img.resize((600, 600))
            self.tk_img = ImageTk.PhotoImage(self.img_resized)

            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Eventos do mouse para seleção
            self.rect = None
            self.start_x = None
            self.start_y = None
            self.crop_coords = None

            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # Salvar posição inicial
        self.start_x = event.x
        self.start_y = event.y

        # Criar retângulo
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
        else:
            self.canvas.coords(self.rect, self.start_x, self.start_y, self.start_x, self.start_y)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)
        # Expandir retângulo
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        # Coordenadas finais
        self.crop_coords = (self.start_x, self.start_y, event.x, event.y)

        # Mapear coordenadas para o tamanho original
        scale_x = self.img.width / 600
        scale_y = self.img.height / 600

        x1 = int(self.start_x * scale_x)
        y1 = int(self.start_y * scale_y)
        x2 = int(event.x * scale_x)
        y2 = int(event.y * scale_y)

        x1 = max(0, min(x1, self.img.width))
        x2 = max(0, min(x2, self.img.width))
        y1 = max(0, min(y1, self.img.height))
        y2 = max(0, min(y2, self.img.height))

        # Recortar e redimensionar a imagem
        crop_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        cropped_img = self.img.crop(crop_box)
        cropped_img_resized = cropped_img.resize((28, 28))

        # Salvar a imagem recortada
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            cropped_img_resized.save(save_path)
            tkinter.messagebox.showinfo("Salvo", f"Imagem recortada salva em {save_path}")

        # Fechar a janela de recorte
        self.crop_window.destroy()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def sidebar_button_event(self):
        print("Botão da barra lateral clicado")

if __name__ == "__main__":
    app = App()
    app.mainloop()