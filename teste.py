import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


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

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Esteatose Hepática",
                                                 font=customtkinter.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.load_image_button = customtkinter.CTkButton(self.sidebar_frame, text="Carregar Imagem ou ROI",
                                                         command=self.load_image, width=200)
        self.load_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Recortar ROI",
                                                        command=self.recortar_roi, width=200)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)
        
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Visualizar Histograma",
                                                        command=self.gerar_histograma, width=200)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Visualizar ROI",
                                                        command=self.sidebar_button_event, width=200)
        self.sidebar_button_2.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Computar GLCM",
                                                        command=self.sidebar_button_event, width=200)
        self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Caracterizar ROI",
                                                        command=self.sidebar_button_event, width=200)
        self.sidebar_button_4.grid(row=5, column=0, padx=20, pady=10)

        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Classificar Imagem",
                                                        command=self.sidebar_button_event, width=200)
        self.sidebar_button_5.grid(row=6, column=0, padx=20, pady=10)

        # Label para exibir a imagem
        self.image_label = customtkinter.CTkLabel(self, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Variáveis iniciais
        self.img = None
        self.img_resized = None
        self.zoom_scale = 1.0  # Variável para controlar o zoom da imagem

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

    def recortar_roi(self):
        # Abrir nova janela para seleção de paciente e exibição de imagem
        self.recorte_window = customtkinter.CTkToplevel(self)
        self.recorte_window.title("Selecionar Paciente e Recortar ROI")
        self.recorte_window.geometry("500x300")
        
        # Configurar layout da nova janela (lista à esquerda e imagem à direita)
        self.recorte_window.grid_columnconfigure(0, weight=1)
        self.recorte_window.grid_columnconfigure(1, weight=4)
        self.recorte_window.grid_rowconfigure(0, weight=1)

        # Criar lista de pacientes à esquerda
        self.patient_listbox = tkinter.Listbox(self.recorte_window, font=("Arial", 14))
        self.patient_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Criar canvas para exibir imagem à direita
        self.canvas_frame = customtkinter.CTkFrame(self.recorte_window)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas = tkinter.Canvas(self.canvas_frame, width=300, height=300)
        self.canvas.pack(fill="both", expand=True)

        # Carregar dados .mat e preencher lista de pacientes
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            data_array = mat_data['data']
            images = data_array['images']

            # Criar uma lista para armazenar todas as imagens e seus índices
            self.image_list = []
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    self.image_list.append((img, patient_idx, img_idx))
                    self.patient_listbox.insert(tkinter.END, f"Paciente {patient_idx}, Imagem {img_idx}")

        # Vincular seleção da lista à exibição da imagem
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_select)

    def on_select(self, event):
        # Carregar a imagem selecionada
        selection = self.patient_listbox.curselection()
        if selection:
            index = selection[0]
            image_data = self.image_list[index][0]

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

            # Redimensionar a imagem para o canvas
            self.img_resized = self.img.resize((600, 600))
            self.tk_img = ImageTk.PhotoImage(self.img_resized)

            # Exibir a imagem no canvas
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Vincular clique para desenhar a ROI
            self.canvas.bind("<Button-1>", self.on_click)

            # Salvar o índice do paciente e da imagem selecionados
            self.selected_patient_idx = self.image_list[index][1]
            self.selected_img_idx = self.image_list[index][2]

    def on_click(self, event):
        # Limpar retângulo anterior, se existir
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)

        # Tamanho do quadrado em pixels na imagem redimensionada
        square_size_display = 28 * (600 / self.img.width)

        # Coordenadas do retângulo na imagem redimensionada
        x1 = event.x - square_size_display / 2
        y1 = event.y - square_size_display / 2
        x2 = event.x + square_size_display / 2
        y2 = event.y + square_size_display / 2

        # Desenhar retângulo verde
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='green')

        # Salvar as coordenadas do clique
        self.click_x = event.x
        self.click_y = event.y

        # Adicionar botão para salvar recorte, se ainda não existir
        if not hasattr(self, 'save_button'):
            self.save_button = customtkinter.CTkButton(self.recorte_window, text="Salvar Recorte", command=self.save_crop)
            self.save_button.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

    def save_crop(self):
        # Mapear coordenadas para o tamanho original
        scale_x = self.img.width / 600
        scale_y = self.img.height / 600

        # Tamanho do quadrado em pixels na imagem original
        square_size_original = 28

        # Coordenadas do retângulo na imagem original
        x_center = self.click_x * scale_x
        y_center = self.click_y * scale_y

        x1 = int(x_center - square_size_original / 2)
        y1 = int(y_center - square_size_original / 2)
        x2 = int(x_center + square_size_original / 2)
        y2 = int(y_center + square_size_original / 2)

        # Garantir que as coordenadas estejam dentro dos limites da imagem
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.img.width, x2)
        y2 = min(self.img.height, y2)

        # Recortar a imagem
        crop_box = (x1, y1, x2, y2)
        cropped_img = self.img.crop(crop_box)

        # Se o recorte não for exatamente 28x28, redimensionar
        if cropped_img.size != (28, 28):
            cropped_img = cropped_img.resize((28, 28))

        # Criar o nome de arquivo padrão
        filename = f"ROI_{self.selected_patient_idx:02d}_{self.selected_img_idx}"

        # Abrir o diálogo "Salvar Como" com o nome de arquivo padrão
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=filename,
            filetypes=[("PNG files", "*.png")]
        )

        if save_path:
            cropped_img.save(save_path)
            tkinter.messagebox.showinfo("Salvo", f"Imagem recortada salva em {save_path}")

    def sidebar_button_event(self):
        print("Botão da barra lateral clicado")
    
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



if __name__ == "__main__":
    app = App()
    app.mainloop()
