import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tkinter.messagebox

class ImageHandler:
    def __init__(self, app):
        self.img = None
        self.img_resized = None
        self.zoom_scale = 1.0  # Variável para controlar o zoom da imagem
        self.image_label = customtkinter.CTkLabel(app, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            self.img_resized = self.img.copy()  # Inicialmente, a imagem redimensionada é a original
            self.display_image()

            # Vincular eventos de rolagem do mouse para o zoom
            self.image_label.bind("<MouseWheel>", self.zoom_image)

    def display_image(self):
        # Redimensionar a imagem conforme o zoom atual
        width, height = self.img.size
        new_size = (int(width * self.zoom_scale), int(height * self.zoom_scale))
        img_resized = self.img.resize(new_size)

        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Manter referência da imagem redimensionada

    def zoom_image(self, event):
        # Ajustar o nível de zoom com base na rolagem do mouse
        if event.delta > 0:
            self.zoom_scale *= 1.1  # Aumentar o zoom em 10%
        elif event.delta < 0:
            self.zoom_scale /= 1.1  # Diminuir o zoom em 10%

        # Limitar o nível de zoom para evitar extremos
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10))  # Limite de 10x e 0.1x

        # Atualizar a exibição da imagem conforme o zoom
        self.display_image()

    def gerar_histograma(self):
        if self.img is None:
            tkinter.messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return

        # Converter a imagem para um array NumPy se ainda não for
        img_array = np.array(self.img)

        # Calcular o histograma com 256 bins (intervalos de pixel de 0 a 255)
        hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

        # Plotar o histograma usando Matplotlib
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Valor do Pixel")
        plt.ylabel("Frequência")
        plt.bar(bins[:-1], hist, width=1, color='gray')  # Gerar o gráfico de barras
        plt.show()