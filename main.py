import customtkinter

from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tkinter.messagebox

import tkinter
from skimage.feature import graycomatrix, graycoprops # Para GLCM
from skimage.util import img_as_ubyte
import os
import tkinter.ttk as ttk
import pandas as pd  # Para operações de CSV
import re # Para RegEx

import scipy.io

import pyfeats
from skimage import io
from tkinter import Tk

SQUARE_SIZE = 28

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.config = AppConfig(self)
        self.image_handler = ImageHandler(self)
        self.roi_handler = ROIHandler(self)
        self.glcm_handler = GLCMHandler(self)
        self.SFM = SFM(self)

    def load_image(self):
        self.image_handler.load_image()

    def gerar_histograma(self):
        self.image_handler.gerar_histograma()

    def recortar_roi(self):
        self.roi_handler.recortar_roi()

    def calcular_hi(self):
        self.roi_handler.calcular_hi_e_ajustar_figado()

    def calcular_glcm(self):
        self.glcm_handler.computar_glcm_roi()

    def sidebar_button_event(self):
        print("Teste botão lateral")

    def calcular_hi_imagem(self):
        self.roi_handler.calcular_hi_imagem()

    def calcular_SFM(self):
        self.SFM.calcular_para_imagem()
    
class AppConfig:
    def __init__(self, app):
        # Definir configuracoes iniciais do menu como titulo e tamanho da janela

        app.title("Diagnóstico de Esteatose Hepática em Exames de Ultrassom")
        app.geometry(f"{1100}x{580}")
        app.grid_columnconfigure(1, weight=1)
        app.grid_rowconfigure((0, 1, 2), weight=1)

        self.create_sidebar(app)

    # Exibir barra lateral do menu
    def create_sidebar(self, app):
        self.sidebar_frame = customtkinter.CTkFrame(app, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=8, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # Botao lateral Esteatose Hepática
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Esteatose Hepática",
            font=customtkinter.CTkFont(size=18, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Botao lateral Carregar Imagem ou ROI
        self.load_image_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Carregar Imagem ou ROI",
            command=app.load_image,
            width=200
        )
        self.load_image_button.grid(row=1, column=0, padx=20, pady=10)
        
        # Botao lateral Visualizar Histograma
        self.visualizar_histograma_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Visualizar Histograma",
            command=app.gerar_histograma,
            width=200
        )
        self.visualizar_histograma_button.grid(row=2, column=0, padx=20, pady=10)

        # Botao lateral Recortar ROI
        self.recortar_roi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Recortar ROI",
            command=app.recortar_roi,
            width=200
        )
        self.recortar_roi_button.grid(row=3, column=0, padx=20, pady=10)

        # Botao lateral Calcular HI
        self.calcular_hi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Calcular HI",
            command=app.calcular_hi_imagem,
            width=200
        )
        self.calcular_hi_button.grid(row=4, column=0, padx=20, pady=10)

        # Botao lateral Computar GLCM
        self.computar_glcm_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Computar GLCM",
            command=app.calcular_glcm,
            width=200
        )
        self.computar_glcm_button.grid(row=5, column=0, padx=20, pady=10)

        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        # Botao lateral SFM
        self.caracterizar_roi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Calcular SFM",
            command=app.calcular_SFM,
            width=200
        )
        self.caracterizar_roi_button.grid(row=6, column=0, padx=20, pady=10)

        # Botao lateral Classificar Imagem
        self.classificar_imagem_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Classificar Imagem (PT02)",
            command=app.sidebar_button_event,
            width=200
        )
        self.classificar_imagem_button.grid(row=7, column=0, padx=20, pady=10)
        
    
class ImageHandler:
    def __init__(self, app):
        self.img = None
        self.img_resized = None
        self.zoom_scale = 1.0  # Variavel para controlar o zoom da img
        self.image_label = customtkinter.CTkLabel(app, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            self.img_resized = self.img.copy()  # Inicialmente, a img redimensionada é a original
            self.display_image()

            # Vincular eventos de rolagem do mouse para o zoom
            self.image_label.bind("<MouseWheel>", self.zoom_image)
    
    # Exibir imgs no menu principal.
    def display_image(self):
        # Redimensionar a img de acordo com zoom
        width, height = self.img.size
        new_size = (int(width * self.zoom_scale), int(height * self.zoom_scale))
        img_resized = self.img.resize(new_size)

        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Manter ref da img redimensionada

    def zoom_image(self, event):
        # Ajustar nivel de zoom com base no scroll
        if event.delta > 0:
            self.zoom_scale *= 1.1  # zoom + 10%
        elif event.delta < 0:
            self.zoom_scale /= 1.1  # zoom - 10%

        # Limitar zoom para evitar extremos
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10)) 

        # Atualizar exibicao conforme zoom
        self.display_image()

    def gerar_histograma(self):
        if self.img is None:
            tkinter.messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return

        # Converter img para um array NumPy
        img_array = np.array(self.img)

        # Calcular histograma com 256 bins (intervalos de pixel de 0 a 255)
        hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

        # Plotar histograma ->  Matplotlib
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Valor do Pixel")
        plt.ylabel("Frequência")
        plt.bar(bins[:-1], hist, width=1, color='gray')
        plt.show()
        
class GLCMHandler:
    def __init__(self, app):
        self.app = app

    def computar_glcm_roi(self):
        roi_path = filedialog.askopenfilename(
            title="Selecione uma ROI",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if roi_path:
            features = self.process_image(roi_path)
            self.display_features(roi_path, features)

    def computar_glcm_roi_directory(self):
        dir_path = filedialog.askdirectory(title="Selecione o diretório com as ROIs")
        if dir_path:
            # Pegar arquivos de imagens
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            # Ordena arquivos com números `zero-padded` na ordem natural
            def natural_sort_key(f):
                # Extrai os segmentos e os ordena a partir dos números
                return [int(num) if num.isdigit() else num for num in re.split(r'(\d+)', f)]
            
            image_files.sort(key=natural_sort_key)

            if not image_files:
                tkinter.messagebox.showinfo("Informação", "Nenhuma imagem encontrada no diretório selecionado.")
                return

            all_features = []
            for image_file in image_files:
                roi_path = os.path.join(dir_path, image_file)
                features = self.process_image(roi_path)
                features['Imagem'] = image_file
                all_features.append(features)

            self.display_features_directory(all_features)
            self.save_features_to_csv(all_features)

    # Calcula a GLCM de uma img de acordo com os ângulos possíveis (de 0 a 360),
    # e de acordo com as distâncias estabelecidas `distances`.
    def process_image(self, roi_path):
        roi_img = Image.open(roi_path).convert("L")
        roi_array = np.array(roi_img)
        roi_array = img_as_ubyte(roi_array)  # Imagem está no formato uint8

        distances = [1, 2, 4, 8]
        levels = 256

        features = {}

        for d in distances:
            angles = np.deg2rad(np.arange(0, 360, 1))  
            # Calula GLCM para todos os ângulos possíveis
            glcm = graycomatrix(
                roi_array,
                distances=[d],
                angles=angles,
                levels=levels,
                symmetric=False,
                normed=True
            )

            # Para cada ângulo, calcule a homogeneidade
            homog = graycoprops(glcm, prop='homogeneity')
            features[f'homogeneity_d{d}'] = np.sum(homog)

            # Entropias
            features[f'entropy_d{d}'] = -np.sum(glcm * np.log2(glcm + 1e-10))

        return features

    def display_features(self, roi_path, features):
        # Nova window para mostrar os features
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM")
        feature_window.geometry("600x400")

        # Display do ROI
        img_frame = customtkinter.CTkFrame(feature_window)
        img_frame.grid(row=0, column=0, padx=10, pady=10)

        img_label = customtkinter.CTkLabel(img_frame, text="ROI Selecionada")
        img_label.pack()

        roi_img = Image.open(roi_path).resize((200, 200))
        roi_photo = ImageTk.PhotoImage(roi_img)
        img_canvas = tkinter.Canvas(img_frame, width=200, height=200)
        img_canvas.pack()
        img_canvas.create_image(0, 0, anchor="nw", image=roi_photo)
        img_canvas.image = roi_photo  # Ref

        # Display das features
        feature_frame = customtkinter.CTkFrame(feature_window)
        feature_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        feature_label = customtkinter.CTkLabel(feature_frame, text="Valores Calculados", font=("Arial", 16))
        feature_label.pack(pady=(0, 10))

        # Listbox para display das features
        listbox = tkinter.Listbox(feature_frame, font=("Arial", 12))
        listbox.pack(fill="both", expand=True)

        for key, value in features.items():
            listbox.insert(tkinter.END, f"{key}: {value:.4f}")

        feature_window.grid_rowconfigure(0, weight=1)
        feature_window.grid_columnconfigure(1, weight=1)

    def display_features_directory(self, all_features):
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM (Diretório)")
        feature_window.geometry("800x600")

        table_frame = customtkinter.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True)

        # Ordem de coluna correta
        columns = [
            'Imagem', 'entropy_d1', 'homogeneity_d1',
            'entropy_d2', 'homogeneity_d2',
            'entropy_d4', 'homogeneity_d4',
            'entropy_d8', 'homogeneity_d8'
        ]

        # Treeview para mostrar dados em formato tabular
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        tree.pack(fill="both", expand=True)

        # Cabeçalho
        for col in columns:
            tree.heading(col, text=col)

        # Dados
        for features in all_features:
            row = [features.get(col, '') for col in columns]
            tree.insert('', tkinter.END, values=row)

        # Scrollbar
        scrollbar = tkinter.Scrollbar(tree, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    def save_features_to_csv(self, all_features):
        # Converte para DataFrame
        df = pd.DataFrame(all_features)

        # Ordem correta de colunas
        columns = [
            'Imagem', 'entropy_d1', 'homogeneity_d1',
            'entropy_d2', 'homogeneity_d2',
            'entropy_d4', 'homogeneity_d4',
            'entropy_d8', 'homogeneity_d8'
        ]

        # Reordena colunas para o esperado
        df = df.reindex(columns=columns, fill_value='')

        # Caso o usuário queira salvar em CSV
        csv_path = filedialog.asksaveasfilename(
            title="Salvar arquivo CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if csv_path:
            df.to_csv(csv_path, index=False)
            tkinter.messagebox.showinfo("Sucesso", f"Arquivo CSV salvo em: {csv_path}")

class ROIHandler:    
    def __init__(self, app):
        self.app = app
        self.selected_patient_idx = None
        self.selected_img_idx = None
        self.img = None
        self.canvas = None
        self.rect = None
        self.click_x = None
        self.click_y = None
        self.save_button = None

    def recortar_roi(self):
        # Abrir nova janela para selecao de paciente e exibicao de img
        self.recorte_window = customtkinter.CTkToplevel()
        self.recorte_window.title("Selecionar Paciente e Recortar ROI")
        self.recorte_window.geometry("1000x600")
        
        # layout da janela 
        self.recorte_window.grid_columnconfigure(0, weight=1)
        self.recorte_window.grid_columnconfigure(1, weight=4)
        self.recorte_window.grid_rowconfigure(0, weight=1)

        # Criar lista de pacientes
        self.patient_listbox = tkinter.Listbox(self.recorte_window, font=("Arial", 14))
        self.patient_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Criar canvas para exibir img
        self.canvas_frame = customtkinter.CTkFrame(self.recorte_window)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas = tkinter.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack(fill="both", expand=True)

        # Carregar dados .mat
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            data_array = mat_data['data']
            images = data_array['images']

            # Criar lista para armazenar todas imgs e indices
            self.image_list = []
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    self.image_list.append((img, patient_idx, img_idx))
                    self.patient_listbox.insert(tkinter.END, f"Paciente {patient_idx}, Imagem {img_idx}")

        # Vincular selecao da lista -> exibicao da img
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_select)

    def on_select(self, event):
        # Carregar a img selecionada
        selection = self.patient_listbox.curselection()
        if selection:
            index = selection[0]
            image_data = self.image_list[index][0]

            # Verificar se image_data e array NumPy
            if not isinstance(image_data, np.ndarray):
                tkinter.messagebox.showerror("Erro", "Dados da imagem invalidos.")
                return

            self.img = Image.fromarray(image_data)

            self.tk_img = ImageTk.PhotoImage(self.img)

            # Exibir img no canvas
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Vincular clique para desenhar ROI
            self.canvas.bind("<Button-1>", self.on_click)

            # Salvar indice do paciente e da img selecionados
            self.selected_patient_idx = self.image_list[index][1]
            self.selected_img_idx = self.image_list[index][2]

    def on_click(self, event):
        # Limpar retangulo anterior
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)

        # Coordenadas do retangulo na img
        x1 = event.x - SQUARE_SIZE // 2
        y1 = event.y - SQUARE_SIZE // 2
        x2 = event.x + SQUARE_SIZE // 2
        y2 = event.y + SQUARE_SIZE // 2

        # Desenhar retangulo na img
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=3)

        # Salvar as coordenadas do clique
        self.click_x = event.x
        self.click_y = event.y

        # Se nao existir, add botao para salvar recorte, s
        if not self.save_button:
            self.save_button = customtkinter.CTkButton(self.recorte_window, text="Salvar Recorte", command=self.save_crop)
            self.save_button.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
            # Add binding da tecla Enter para salvar recorte
            self.recorte_window.bind('<Return>', lambda event: self.save_crop())
    
    def save_crop(self):

        # Coordenadas do retangulo na img original
        x1 = self.click_x - (SQUARE_SIZE // 2)
        y1 = self.click_y - (SQUARE_SIZE // 2)
        x2 = self.click_x + (SQUARE_SIZE // 2)
        y2 = self.click_y + (SQUARE_SIZE // 2)

        # Garantir que coordenadas estao dentro dos limites da img
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.img.width, x2)
        y2 = min(self.img.height, y2)

        # Recortar img
        crop_box = (x1, y1, x2, y2)
        cropped_img = self.img.crop(crop_box)

        # Criar nome de arquivo padrao -> ROI_nn_mm
        filename = f"ROI_{self.selected_patient_idx:02d}_{self.selected_img_idx}"

        # Abrir dialogo de "Salvar Como" com nome de arquivo padrao
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=filename,
            filetypes=[("PNG files", "*.png")]
        )

        if save_path:
            # Salvar img recortada
            cropped_img.save(save_path)

            # Salvar coordenadas ROI em um arquivo de texto compartilhado
            coord_save_path = "ROI_coordinates.txt"
            with open(coord_save_path, 'a') as f:
                f.write(f"{filename}: Coordinates of top-left corner: (x={x1}, y={y1})\n")

            tkinter.messagebox.showinfo("Salvo", f"Imagem recortada salva em {save_path}\nCoordenadas salvas em {coord_save_path}")

    def calcular_hi_e_ajustar_figado(self):
        # Diretorios das ROIs
        figado_dir = "./figado"
        rim_dir = "./rim"
        output_dir = "./Figado_Ajustado"
        os.makedirs(output_dir, exist_ok=True)

        # Arquivo para salvar valores de HI
        hi_save_path = "HI_values.txt"

        # Iterar sobre todas ROIs do figado
        with open(hi_save_path, 'w') as hi_file:
            for filename in os.listdir(figado_dir):
                if filename.endswith(".png") and "RIM" not in filename:
                    # Carregar ROI do figado e do rim correspondente
                    figado_path = os.path.join(figado_dir, filename)
                    rim_path = os.path.join(rim_dir, filename.replace('.png', '_RIM.png'))

                    if not os.path.exists(rim_path):
                        continue

                    figado_img = Image.open(figado_path).convert("L")
                    rim_img = Image.open(rim_path).convert("L")

                    # Converter imgs para arrays NumPy
                    figado_array = np.array(figado_img)
                    rim_array = np.array(rim_img)

                    # Calcular a media dos tons de cinza
                    media_figado = np.mean(figado_array)
                    media_rim = np.mean(rim_array)

                    # Calcular o HI e salvar no arquivo
                    hi = media_figado / media_rim if media_rim != 0 else 1
                    hi_file.write(f"{filename.replace('.png', '')}, {hi}\n")

                    # Ajustar tons de cinza da ROI do figado
                    ajustado_figado_array = np.clip(np.round(figado_array * hi), 0, 255).astype(np.uint8)

                    # Criar img ajustada do figado
                    ajustado_figado_img = Image.fromarray(ajustado_figado_array)

                    # Salvar ROI ajustada do figado
                    ajustado_figado_path = os.path.join(output_dir, filename)
                    ajustado_figado_img.save(ajustado_figado_path)

        tkinter.messagebox.showinfo("Salvo", f"Imagens ajustadas e salvas em {output_dir}\n valores do HI salvos em {hi_save_path}")


    def calcular_hi_imagem(self):
        # Abrir nova janela para selecao de paciente e exibicao de img
        self.hi_window = customtkinter.CTkToplevel()
        self.hi_window.title("Selecionar Paciente e Calcular HI")
        self.hi_window.geometry("1000x600")
        
        # Configurar layout da nova janela
        self.hi_window.grid_columnconfigure(0, weight=1)
        self.hi_window.grid_columnconfigure(1, weight=4)
        self.hi_window.grid_rowconfigure(0, weight=1)

        # Criar lista de pacientes 
        self.patient_listbox = tkinter.Listbox(self.hi_window, font=("Arial", 14))
        self.patient_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Criar canvas para exibir img 
        self.canvas_frame = customtkinter.CTkFrame(self.hi_window)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas = tkinter.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack(fill="both", expand=True)

        # Carregar dados .mat e preencher lista de pacientes
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            data_array = mat_data['data']
            images = data_array['images']

            # Criar uma lista para armazenar todas as imgs e seus indices
            self.image_list = []
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    self.image_list.append((img, patient_idx, img_idx))
                    self.patient_listbox.insert(tkinter.END, f"Paciente {patient_idx}, Imagem {img_idx}")

        # Vincular selecao da lista a exibicao da img
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_select_hi)

    def on_select_hi(self, event):
        # Carregar a img selecionada
        selection = self.patient_listbox.curselection()
        if selection:
            index = selection[0]
            image_data = self.image_list[index][0]

            # Verificar se image_data e um array NumPy
            if not isinstance(image_data, np.ndarray):
                tkinter.messagebox.showerror("Erro", "Dados da imagem invalidos.")
                return

            self.img = Image.fromarray(image_data).convert("L")

            self.tk_img = ImageTk.PhotoImage(self.img)

            # Limpar qualquer conteudo anterior do canvas
            self.canvas.delete("all")

            # Exibir img no canvas
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Vincular clique para selecionar pontos
            self.canvas.bind("<Button-1>", self.on_click_hi)

            # Salvar indice do paciente e da img selecionados
            self.selected_patient_idx = self.image_list[index][1]
            self.selected_img_idx = self.image_list[index][2]

            # Inicializar variaveis para selecao de pontos
            self.points = []
            self.rects = []

    def on_click_hi(self, event):
        if len(self.rects) == 2: return 
        # Desenhar retangulo no ponto clicado
        x1 = event.x - SQUARE_SIZE // 2
        y1 = event.y - SQUARE_SIZE // 2
        x2 = event.x + SQUARE_SIZE // 2
        y2 = event.y + SQUARE_SIZE // 2


        if not len(self.rects) == 2:
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
            self.rects.append(rect)
            # Salvar coordenadas do clique
            self.points.append((event.x, event.y))


        if len(self.points) == 2:
            # Apos dois pontos selecionados calcular o HI
            self.calculate_hi_from_points()
            # Resetar os pontos e retangulos para permitir novas selecoes

    def calculate_hi_from_points(self):
        # Obter coordenadas dos dois pontos
        (x1, y1) = self.points[0]
        (x2, y2) = self.points[1]

        # Coordenadas na img original
        x1_original = x1
        y1_original = y1
        x2_original = x2
        y2_original = y2

        # Definir caixas ao redor dos pontos
        box1 = (
            x1_original - SQUARE_SIZE // 2,
            y1_original - SQUARE_SIZE // 2,
            x1_original + SQUARE_SIZE // 2,
            y1_original + SQUARE_SIZE // 2
        )
        box2 = (
            x2_original - SQUARE_SIZE // 2,
            y2_original - SQUARE_SIZE // 2,
            x2_original + SQUARE_SIZE // 2,
            y2_original + SQUARE_SIZE // 2
        )

        # Garantir que coordenadas estejam dentro dos limites da img
        box1 = (
            max(0, box1[0]),
            max(0, box1[1]),
            min(self.img.width, box1[2]),
            min(self.img.height, box1[3])
        )
        box2 = (
            max(0, box2[0]),
            max(0, box2[1]),
            min(self.img.width, box2[2]),
            min(self.img.height, box2[3])
        )

        # Recortar regioes
        region1 = self.img.crop(box1)
        region2 = self.img.crop(box2)

        # Converter regioes para arrays NumPy
        array1 = np.array(region1)
        array2 = np.array(region2)

        # Calcular medias das regioes
        media1 = np.mean(array1)
        media2 = np.mean(array2)

        # Calcular HI (media2 -> rim)
        hi = media1 / media2 if media2 != 0 else 1

        # Exibir valor de HI ao user
        tkinter.messagebox.showinfo("HI Calculado", f"O valor do HI e: {hi:.4f}")

        # Ajustar tons de cinza da ROI do figado
        ajustado_array = np.clip(np.round(array1 * hi), 0, 255).astype(np.uint8)

        # Criar img ajustada do figado
        ajustado_region1 = Image.fromarray(ajustado_array)

        # Diretorios para salvar as ROIs
        figado_dir = "./figadoTeste"
        figado_ajustado_dir = "./Figado_AjustadoTeste"
        os.makedirs(figado_dir, exist_ok=True)
        os.makedirs(figado_ajustado_dir, exist_ok=True)

        # Criar nomes de arquivo
        filename_liver_roi = f"ROI_FIGADO_{self.selected_patient_idx:02d}_{self.selected_img_idx}.png"
        filename_liver_roi_ajustado = f"ROI_FIGADO_AJUSTADO_{self.selected_patient_idx:02d}_{self.selected_img_idx}.png"

        # Caminhos completos
        save_path_liver = os.path.join(figado_dir, filename_liver_roi)
        save_path_liver_ajustado = os.path.join(figado_ajustado_dir, filename_liver_roi_ajustado)

        # Salvar imgs e informar user
        region1.save(save_path_liver)
        ajustado_region1.save(save_path_liver_ajustado)
        tkinter.messagebox.showinfo("Salvo", f"ROI do figado original salva em {save_path_liver}\nROI do figado ajustada salva em {save_path_liver_ajustado}")

        # Apagar retangulos desenhados no canvas
        for rect in self.rects:
            self.canvas.delete(rect)

        # Resetar listas de pontos e retangulos
        self.points = []
        self.rects = []

class SFM:

    def __init__(self, app, Lr=4, Lc=4):
        self.app = app
        self.Lr = Lr
        self.Lc = Lc
    
    def calcular_para_imagem(self, mask=None):
        filepath = filedialog.askopenfilename(
            title="Selecione a imagem",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not filepath:
            print("Seleção de arquivo cancelada pelo usuário.")
            return
        
        f = io.imread(filepath, as_gray=True)
        features, labels = pyfeats.sfm_features(f, mask, self.Lr, self.Lc)
        
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Resultados das Características SFM")
        feature_window.geometry("1000x700")

        table_frame = customtkinter.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True, side="left", padx=10, pady=10)

        text_widget = customtkinter.CTkTextbox(table_frame, wrap="word")
        text_widget.pack(fill="both", expand=True)

        text_widget.insert("end", f"Nome do Arquivo: {os.path.basename(filepath)}\n\n")
        text_widget.insert("end", "Características:\n")
        
        for label, feature in zip(labels, features):
            text_widget.insert("end", f"{label}: {feature}\n")
        
        text_widget.configure(state="disabled")
        
        img_frame = customtkinter.CTkFrame(feature_window, width=500, height=500)
        img_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        pil_img = Image.open(filepath)
        pil_img = pil_img.resize((500, 500), Image.LANCZOS)  # Redimensiona a img para 500x500
        self.tk_img = ImageTk.PhotoImage(pil_img)  # Armazena a img como atributo da instância
        
        img_label = customtkinter.CTkLabel(img_frame, image=self.tk_img, text="")  # Remove o texto padrão
        img_label.pack(expand=True)

    def calcular_para_pasta(self):
        # Abre uma janela para o usuário selecionar o diretório com as imagens
        root = Tk()
        root.withdraw()  # Esconde a janela principal do Tkinter
        directory = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        
        # Se o usuário cancelar a seleção da pasta, interrompe a execução
        if not directory:
            print("Seleção de pasta cancelada pelo usuário.")
            return
        
        data = []  # Lista para armazenar resultados
        
        # Loop em todos os arquivos no diretório
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):  # Filtra para arquivos de img
                filepath = os.path.join(directory, filename)
                
                # Carrega a img e considera a img inteira como ROI
                f = io.imread(filepath, as_gray=True)
                features, labels = pyfeats.sfm_features(f, None, self.Lr, self.Lc)
                
                # Adiciona os resultados à lista de dados
                row = [filename] + features.tolist()
                data.append(row)

        # Cria um DataFrame com os resultados
        columns = ['Filename'] + labels
        df = pd.DataFrame(data, columns=columns)
        
        # Abre uma janela para o usuário escolher o local de salvamento do arquivo CSV
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="resultados_sfm_features.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        # Salva o DataFrame em um arquivo CSV se o usuário escolher um caminho
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"CSV gerado com sucesso em: {save_path}")
        else:
            print("Salvamento cancelado pelo usuário.")
            
if __name__ == "__main__":
    app = App()
    app.mainloop()