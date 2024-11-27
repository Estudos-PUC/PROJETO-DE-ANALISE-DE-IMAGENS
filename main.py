# Copyright(c) Eduardo Lemos, Fernanda Gomes, Pedro Olyntho - 2024

#=========================================
# Imports.
#=========================================

# Biblioteca Padrão.
import os
import re
import sys
import math
import time
from tkinter import Tk, filedialog, messagebox, ttk

# Manipular dados.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# Processamento de Imagens
from PIL import Image, ImageTk
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
import pyfeats

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# DeepLearning Frameworks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# GUI
import customtkinter

#=========================================
# Constantes Globais.
#=========================================

SQUARE_SIZE = 28

#=========================================
# Classe Principal do Aplicativo.
#=========================================

class App(customtkinter.CTk):
    """
    Configurar a interface principal do programa e inicializar manipuladores das imgs.
    
    Metodos: 
        load_image
        gerar_histograma
        recortar_roi
        calcular_hi
        calcular_glcm
        sidebar_button_event
        calcular_hi_imagem
        calcular_SFM
        classificar_imagem_SVM
    """
    def __init__(self):
        super().__init__()
        self.config = AppConfig(self)
        self.image_handler = ImageHandler(self)
        self.roi_handler = ROIHandler(self)
        self.glcm_handler = GLCMHandler(self)
        self.SFM = SFM(self)
        self.SVMClassifier = SVMClassifier(self)
        self.Resnet50 = Resnet50(self)

    def load_image(self):
        self.SVMClassifier.hide_metrics()
        self.image_handler.show_label()
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
        print("Teste botao lateral")

    def calcular_hi_imagem(self):
        self.roi_handler.calcular_hi_imagem()

    def calcular_SFM(self):
        self.SFM.calcular_para_imagem()
    
    def classificar_imagem_SVM(self):
        self.image_handler.hide_label()
        self.SVMClassifier.load_data()
        self.SVMClassifier.validate()
        self.SVMClassifier.plot_confusion_matrix()
        # self.SVMClassifier.predict_image()
        self.SVMClassifier.calculate_metrics()
        self.SVMClassifier.show_metrics()

    def classificar_imagem_Resnet50(self):
        self.image_handler.hide_label()
        self.Resnet50.load_model()
        self.Resnet50.classify_single_image()

    def train_Resnet50(self):
        self.Resnet50.select_folder()
        self.Resnet50.run()

#=========================================
# Configuração da Interface (AppConfig).
#=========================================

class AppConfig:
    """
    Criar janela do programa, configurando a barra lateral com os botoes para manipulacao das imgs.
    
    Metodos:
        create_sidebar
    """
    def __init__(self, app):
        # Definir configuracoes iniciais do menu 
        app.title("Diagnostico de Esteatose Hepatica em Exames de Ultrassom")
        app.geometry(f"{1100}x{580}")
        app.grid_columnconfigure(1, weight=1)
        app.grid_rowconfigure((0, 1, 2), weight=1)

        self.create_sidebar(app)

    # Exibir barra lateral do menu
    def create_sidebar(self, app):
        self.sidebar_frame = customtkinter.CTkFrame(app, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        # Botao lateral Esteatose Hepatica
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Esteatose Hepatica",
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

        # Botao lateral SFM
        self.caracterizar_roi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Calcular SFM",
            command=app.calcular_SFM,
            width=200
        )
        self.caracterizar_roi_button.grid(row=6, column=0, padx=20, pady=10)

        # Botao lateral Classificar Imagem
        self.SVM_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="SVM",
            command=app.classificar_imagem_SVM,
            width=200
        )
        self.SVM_button.grid(row=7, column=0, padx=20, pady=10)

        # Botao lateral Classificar Imagem
        self.Resnet50_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Resnet50",
            command=app.classificar_imagem_Resnet50,
            width=200
        )
        self.Resnet50_button.grid(row=8, column=0, padx=20, pady=10)

        # Botao lateral Treinar ResNet50
        self.TrainResnet50_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Treinar Resnet50",
            command=app.train_Resnet50,
            width=200
        )
        self.TrainResnet50_button.grid(row=9, column=0, padx=20, pady=10)
        
#=========================================
# Manipulação de Imagens (ImageHandler).
#=========================================

class ImageHandler:
    """
    Carregar a imagem na janela principal, dar zoom na imagem carregada e calcular o histograma.
    
    Metodos: 
        load_image
        display_image
        zoom_image
        gerar_histograma
    """
    def __init__(self, app):
        self.img = None
        self.img_resized = None
        self.zoom_scale = 1.0  # Variavel para controlar o zoom da img.
        self.image_label = customtkinter.CTkLabel(app, text="Nenhuma Imagem Carregada")
        self.image_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            self.img_resized = self.img.copy()  # Inicialmente, a img redimensionada e a original
            self.display_image()

            # Vincular eventos de rolagem do mouse para o zoom.
            self.image_label.bind("<MouseWheel>", self.zoom_image)
    
    # Exibir imgs no menu principal.
    def display_image(self):
        # Redimensionar a img de acordo com zoom.
        width, height = self.img.size
        new_size = (int(width * self.zoom_scale), int(height * self.zoom_scale))
        img_resized = self.img.resize(new_size)

        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Manter ref da img redimensionada.

    def zoom_image(self, event):
        # Ajustar nivel de zoom com base no scroll.
        if event.delta > 0:
            self.zoom_scale *= 1.1  # zoom + 10%
        elif event.delta < 0:
            self.zoom_scale /= 1.1  # zoom - 10%

        # Limitar zoom para evitar extremos.
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10)) 

        # Atualizar exibicao conforme zoom.
        self.display_image()

    def gerar_histograma(self):
        if self.img is None:
            tkinter.messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return

        img_array = np.array(self.img)

        # Calcular histograma com 256 bins (intervalos de pixel de 0 a 255).
        hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

        # Plotar histograma ->  Matplotlib
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Valor do Pixel")
        plt.ylabel("Frequência")
        plt.bar(bins[:-1], hist, width=1, color='gray')
        plt.show()

    def show_label(self):
        self.image_label.grid()

    def hide_label(self):
        self.image_label.grid_remove()

#=========================================
# Matriz de Co-ocorrência de Níveis de Cinza.
#=========================================

class GLCMHandler:
    """
    Computar e exibir GLCM  de uma ROI ou de todas as ROIs. 
    
    Metodos: 
        computar_glcm_roi
        computar_glcm_roi_directory
        process_image
        display_features
        display_features_directory
        save_features_to_csv
    """
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
        dir_path = filedialog.askdirectory(title="Selecione o diretorio com as ROIs")
        if dir_path:
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            # Ordena arquivos com números `zero-padded` na ordem natural.
            def natural_sort_key(f):
                # Extrai os segmentos e os ordena a partir dos números.
                return [int(num) if num.isdigit() else num for num in re.split(r'(\d+)', f)]
            
            image_files.sort(key=natural_sort_key)

            if not image_files:
                tkinter.messagebox.showinfo("Informacao", "Nenhuma imagem encontrada no diretorio selecionado.")
                return

            all_features = []
            for image_file in image_files:
                roi_path = os.path.join(dir_path, image_file)
                features = self.process_image(roi_path)
                features['Imagem'] = image_file
                all_features.append(features)

            self.display_features_directory(all_features)
            self.save_features_to_csv(all_features)

    # Calcular a GLCM de uma img de acordo com os angulos possiveis (de 0 a 360),
    # e de acordo com as distancias estabelecidas.
    def process_image(self, roi_path):
        roi_img = Image.open(roi_path).convert("L")
        roi_array = np.array(roi_img)
        roi_array = img_as_ubyte(roi_array)  # Imagem no formato uint8.

        distances = [1, 2, 4, 8]
        levels = 256

        features = {}

        for d in distances:
            angles = np.deg2rad(np.arange(0, 360, 1))  
            # Calula GLCM para todos os angulos possiveis.
            glcm = graycomatrix(
                roi_array,
                distances=[d],
                angles=angles,
                levels=levels,
                symmetric=False,
                normed=True
            )
            # Entropias
            features[f'entropy_d{d}'] = -np.sum(glcm * np.log2(glcm + 1e-10))

            # Para cada angulo, calcule a homogeneidade.
            homog = graycoprops(glcm, prop='homogeneity')
            features[f'homogeneity_d{d}'] = np.sum(homog)



        return features

    def display_features(self, roi_path, features):
        # Nova window para mostrar os features.
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM")
        feature_window.geometry("600x400")

        # Display do ROI.
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

        # Display das features.
        feature_frame = customtkinter.CTkFrame(feature_window)
        feature_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        feature_label = customtkinter.CTkLabel(feature_frame, text="Valores Calculados", font=("Arial", 16))
        feature_label.pack(pady=(0, 10))

        # Listbox para display das features.
        listbox = tkinter.Listbox(feature_frame, font=("Arial", 12))
        listbox.pack(fill="both", expand=True)

        for key, value in features.items():
            listbox.insert(tkinter.END, f"{key}: {value:.4f}")

        feature_window.grid_rowconfigure(0, weight=1)
        feature_window.grid_columnconfigure(1, weight=1)

    def display_features_directory(self, all_features):
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM (Diretorio)")
        feature_window.geometry("800x600")

        table_frame = customtkinter.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True)

        # Ordem de coluna correta.
        columns = [
            'Imagem', 'entropy_d1', 'homogeneity_d1',
            'entropy_d2', 'homogeneity_d2',
            'entropy_d4', 'homogeneity_d4',
            'entropy_d8', 'homogeneity_d8'
        ]

        # Treeview para mostrar dados em formato tabular.
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        tree.pack(fill="both", expand=True)

        # Cabecalho.
        for col in columns:
            tree.heading(col, text=col)

        # Dados.
        for features in all_features:
            row = [features.get(col, '') for col in columns]
            tree.insert('', tkinter.END, values=row)

        # Scrollbar.
        scrollbar = tkinter.Scrollbar(tree, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    def save_features_to_csv(self, all_features):
        df = pd.DataFrame(all_features)

        # Ordem correta de colunas.
        columns = [
            'Imagem', 'entropy_d1', 'homogeneity_d1',
            'entropy_d2', 'homogeneity_d2',
            'entropy_d4', 'homogeneity_d4',
            'entropy_d8', 'homogeneity_d8'
        ]

        # Reordena colunas para o esperado.
        df = df.reindex(columns=columns, fill_value='')

        # Opcao de salvar em CSV.
        csv_path = filedialog.asksaveasfilename(
            title="Salvar arquivo CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if csv_path:
            df.to_csv(csv_path, index=False)
            tkinter.messagebox.showinfo("Sucesso", f"Arquivo CSV salvo em: {csv_path}")

#=========================================
# Manipulação de ROIs.
#=========================================

class ROIHandler:    

    """
    Recortar ROI e calcular HI

    
    Mestodos:
        recortar_roi
        on_select
        on_click
        save_crop
        calcular_hi_e_ajustar_figado
        calcular_hi_imagem
        on_select_hi
        on_click_hi
        calculate_hi_from_points
    """

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
        # Abrir nova janela para selecao de paciente e exibicao de img.
        self.recorte_window = customtkinter.CTkToplevel()
        self.recorte_window.title("Selecionar Paciente e Recortar ROI")
        self.recorte_window.geometry("1000x600")
        
        # Layout da janela.
        self.recorte_window.grid_columnconfigure(0, weight=1)
        self.recorte_window.grid_columnconfigure(1, weight=4)
        self.recorte_window.grid_rowconfigure(0, weight=1)

        # Lista de pacientes.
        self.patient_listbox = tkinter.Listbox(self.recorte_window, font=("Arial", 14))
        self.patient_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Canvas para exibir img.
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

            # Lista para armazenar todas imgs e indices.
            self.image_list = []
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    self.image_list.append((img, patient_idx, img_idx))
                    self.patient_listbox.insert(tkinter.END, f"Paciente {patient_idx}, Imagem {img_idx}")

        # Vincular selecao da lista -> exibicao da img.
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_select)

    def on_select(self, event):
        # Carregar img selecionada.
        selection = self.patient_listbox.curselection()
        if selection:
            index = selection[0]
            image_data = self.image_list[index][0]

            if not isinstance(image_data, np.ndarray):
                tkinter.messagebox.showerror("Erro", "Dados da imagem invalidos.")
                return

            self.img = Image.fromarray(image_data)

            self.tk_img = ImageTk.PhotoImage(self.img)

            # Exibir img no canvas.
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Vincular clique para desenhar ROI.
            self.canvas.bind("<Button-1>", self.on_click)

            # Salvar indice do paciente e da img selecionados.
            self.selected_patient_idx = self.image_list[index][1]
            self.selected_img_idx = self.image_list[index][2]

    def on_click(self, event):
        # Limpar retangulo anterior.
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)

        # Coordenadas do retangulo na img.
        x1 = event.x - SQUARE_SIZE // 2
        y1 = event.y - SQUARE_SIZE // 2
        x2 = event.x + SQUARE_SIZE // 2
        y2 = event.y + SQUARE_SIZE // 2

        # Desenhar retangulo na img.
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=3)

        # Salvar as coordenadas do clique.
        self.click_x = event.x
        self.click_y = event.y

        # Se nao existir, add botao para salvar recorte.
        if not self.save_button:
            self.save_button = customtkinter.CTkButton(self.recorte_window, text="Salvar Recorte", command=self.save_crop)
            self.save_button.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
            # Binding em Enter para salvar recorte.
            self.recorte_window.bind('<Return>', lambda event: self.save_crop())
    
    def save_crop(self):

        # Coordenadas do retangulo na img original.
        x1 = self.click_x - (SQUARE_SIZE // 2)
        y1 = self.click_y - (SQUARE_SIZE // 2)
        x2 = self.click_x + (SQUARE_SIZE // 2)
        y2 = self.click_y + (SQUARE_SIZE // 2)

        # Garantir que coordenadas estao dentro dos limites da img.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.img.width, x2)
        y2 = min(self.img.height, y2)

        # Recortar img.
        crop_box = (x1, y1, x2, y2)
        cropped_img = self.img.crop(crop_box)

        # Criar nome de arquivo padrao -> ROI_nn_mm.
        filename = f"ROI_{self.selected_patient_idx:02d}_{self.selected_img_idx}"

        # Abrir dialogo de "Salvar Como" com nome de arquivo padrao.
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=filename,
            filetypes=[("PNG files", "*.png")]
        )

        if save_path:
            cropped_img.save(save_path)

            # Salvar coordenadas ROI em um arquivo de texto compartilhado.
            coord_save_path = "ROI_coordinates.txt"
            with open(coord_save_path, 'a') as f:
                f.write(f"{filename}: Coordinates of top-left corner: (x={x1}, y={y1})\n")

            tkinter.messagebox.showinfo("Salvo", f"Imagem recortada salva em {save_path}\nCoordenadas salvas em {coord_save_path}")

    def calcular_hi_e_ajustar_figado(self):
        # Diretorios das ROIs.
        figado_dir = "./figado"
        rim_dir = "./rim"
        output_dir = "./Figado_Ajustado"
        os.makedirs(output_dir, exist_ok=True)

        # Arquivo para salvar valores de HI.
        hi_save_path = "HI_values.txt"

        # Iterar sobre todas ROIs do figado.
        with open(hi_save_path, 'w') as hi_file:
            for filename in os.listdir(figado_dir):
                if filename.endswith(".png") and "RIM" not in filename:
                    # Carregar ROI do figado e do rim correspondente.
                    figado_path = os.path.join(figado_dir, filename)
                    rim_path = os.path.join(rim_dir, filename.replace('.png', '_RIM.png'))

                    if not os.path.exists(rim_path):
                        continue

                    figado_img = Image.open(figado_path).convert("L")
                    rim_img = Image.open(rim_path).convert("L")

                    figado_array = np.array(figado_img)
                    rim_array = np.array(rim_img)

                    media_figado = np.mean(figado_array)
                    media_rim = np.mean(rim_array)

                    # Calcular o HI e salvar no arquivo.
                    hi = media_figado / media_rim if media_rim != 0 else 1
                    hi_file.write(f"{filename.replace('.png', '')}, {hi}\n")

                    # Ajustar tons de cinza da ROI do figado.
                    ajustado_figado_array = np.clip(np.round(figado_array * hi), 0, 255).astype(np.uint8)

                    # Criar img ajustada do figado.
                    ajustado_figado_img = Image.fromarray(ajustado_figado_array)

                    # Salvar ROI ajustada do figado.
                    ajustado_figado_path = os.path.join(output_dir, filename)
                    ajustado_figado_img.save(ajustado_figado_path)

        tkinter.messagebox.showinfo("Salvo", f"Imagens ajustadas e salvas em {output_dir}\n valores do HI salvos em {hi_save_path}")


    def calcular_hi_imagem(self):
        # Abrir nova janela para selecao de paciente e exibicao de img.
        self.hi_window = customtkinter.CTkToplevel()
        self.hi_window.title("Selecionar Paciente e Calcular HI")
        self.hi_window.geometry("1000x600")
        
        # Configurar layout da nova janela.
        self.hi_window.grid_columnconfigure(0, weight=1)
        self.hi_window.grid_columnconfigure(1, weight=4)
        self.hi_window.grid_rowconfigure(0, weight=1)

        # Lista de pacientes.
        self.patient_listbox = tkinter.Listbox(self.hi_window, font=("Arial", 14))
        self.patient_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Canvas para exibir img.
        self.canvas_frame = customtkinter.CTkFrame(self.hi_window)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas = tkinter.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack(fill="both", expand=True)

        # Carregar dados .mat e preencher lista de pacientes.
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            data_array = mat_data['data']
            images = data_array['images']

            # Lista para armazenar todas as imgs e seus indices.
            self.image_list = []
            for patient_idx in range(images.shape[1]):
                patient_images = images[0, patient_idx]
                for img_idx in range(len(patient_images)):
                    img = patient_images[img_idx]
                    self.image_list.append((img, patient_idx, img_idx))
                    self.patient_listbox.insert(tkinter.END, f"Paciente {patient_idx}, Imagem {img_idx}")

        # Vincular selecao da lista a exibicao da img.
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_select_hi)

    def on_select_hi(self, event):
        # Carregar a img selecionada.
        selection = self.patient_listbox.curselection()
        if selection:
            index = selection[0]
            image_data = self.image_list[index][0]

            # Verificar se image_data e um array NumPy.
            if not isinstance(image_data, np.ndarray):
                tkinter.messagebox.showerror("Erro", "Dados da imagem invalidos.")
                return

            self.img = Image.fromarray(image_data).convert("L")

            self.tk_img = ImageTk.PhotoImage(self.img)

            # Limpar qualquer conteudo anterior do canvas.
            self.canvas.delete("all")

            # Exibir img no canvas.
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Bind clique para selecionar pontos.
            self.canvas.bind("<Button-1>", self.on_click_hi)

            # Salvar indice do paciente e da img selecionados.
            self.selected_patient_idx = self.image_list[index][1]
            self.selected_img_idx = self.image_list[index][2]

            self.points = []
            self.rects = []

    def on_click_hi(self, event):
        if len(self.rects) == 2: return 
        # Desenhar retangulo no ponto clicado.
        x1 = event.x - SQUARE_SIZE // 2
        y1 = event.y - SQUARE_SIZE // 2
        x2 = event.x + SQUARE_SIZE // 2
        y2 = event.y + SQUARE_SIZE // 2


        if not len(self.rects) == 2:
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
            self.rects.append(rect)
            # Salvar coordenadas do clique.
            self.points.append((event.x, event.y))


        if len(self.points) == 2:
            # Apos dois pontos selecionados calcular o HI.
            self.calculate_hi_from_points()
            # Resetar os pontos e retangulos para permitir novas selecoes.

    def calculate_hi_from_points(self):
        (x1, y1) = self.points[0]
        (x2, y2) = self.points[1]

        # Coordenadas na img original.
        x1_original = x1
        y1_original = y1
        x2_original = x2
        y2_original = y2

        # Definir caixas ao redor dos pontos.
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

        # Garantir que coordenadas estejam dentro dos limites da img.
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

        # Recortar regioes.
        region1 = self.img.crop(box1)
        region2 = self.img.crop(box2)

        # Converter regioes para arrays NumPy.
        array1 = np.array(region1)
        array2 = np.array(region2)

        # Calcular medias das regioes.
        media1 = np.mean(array1)
        media2 = np.mean(array2)

        # Calcular HI (media2 -> rim).
        hi = media1 / media2 if media2 != 0 else 1

        # Exibir valor de HI ao user.
        tkinter.messagebox.showinfo("HI Calculado", f"O valor do HI e: {hi:.4f}")

        # Ajustar tons de cinza da ROI do figado.
        ajustado_array = np.clip(np.round(array1 * hi), 0, 255).astype(np.uint8)

        # img ajustada do figado.
        ajustado_region1 = Image.fromarray(ajustado_array)

        # Diretorios para salvar as ROIs.
        figado_dir = "./figadoTeste"
        figado_ajustado_dir = "./Figado_AjustadoTeste"
        os.makedirs(figado_dir, exist_ok=True)
        os.makedirs(figado_ajustado_dir, exist_ok=True)

        # Criar nomes de arquivo.
        filename_liver_roi = f"ROI_FIGADO_{self.selected_patient_idx:02d}_{self.selected_img_idx}.png"
        filename_liver_roi_ajustado = f"ROI_FIGADO_AJUSTADO_{self.selected_patient_idx:02d}_{self.selected_img_idx}.png"

        # Caminhos completos.
        save_path_liver = os.path.join(figado_dir, filename_liver_roi)
        save_path_liver_ajustado = os.path.join(figado_ajustado_dir, filename_liver_roi_ajustado)

        # Salvar imgs e informar user.
        region1.save(save_path_liver)
        ajustado_region1.save(save_path_liver_ajustado)
        tkinter.messagebox.showinfo("Salvo", f"ROI do figado original salva em {save_path_liver}\nROI do figado ajustada salva em {save_path_liver_ajustado}")

        # Apagar retangulos desenhados no canvas.
        for rect in self.rects:
            self.canvas.delete(rect)

        # Resetar listas de pontos e retangulos.
        self.points = []
        self.rects = []

#=========================================
# Cálculo do SFM.
#=========================================

class SFM:
    """
    Calcular os valores do SFM de apenas uma imagem e calcular de todas as imagens de uma pasta

    Essa é a principal funcao e utilizamos a biblioteca pyfeats:
    features, labels = pyfeats.sfm_features(f, mask, self.Lr, self.Lc)

    f: imagem selecionada
    mask: por padrao none porque a imagem selecionada ja e uma roi

    valores retirados da documentacao da biblioteca
    self.Lr
    self.lc


    Metodos
        calcular_para_imagem
        calcular_para_pasta
    """


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
            print("Selecao de arquivo cancelada pelo usuario.")
            return
        
        f = io.imread(filepath, as_gray=True)
        features, labels = pyfeats.sfm_features(f, mask, self.Lr, self.Lc)
        
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Resultados das Caracteristicas SFM")
        feature_window.geometry("1000x700")

        table_frame = customtkinter.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True, side="left", padx=10, pady=10)

        text_widget = customtkinter.CTkTextbox(table_frame, wrap="word")
        text_widget.pack(fill="both", expand=True)

        text_widget.insert("end", f"Nome do Arquivo: {os.path.basename(filepath)}\n\n")
        text_widget.insert("end", "Caracteristicas:\n")
        
        for label, feature in zip(labels, features):
            text_widget.insert("end", f"{label}: {feature}\n")
        
        text_widget.configure(state="disabled")
        
        img_frame = customtkinter.CTkFrame(feature_window, width=500, height=500)
        img_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        pil_img = Image.open(filepath)
        pil_img = pil_img.resize((500, 500), Image.LANCZOS)  # Redimensiona a img para 500x500
        self.tk_img = ImageTk.PhotoImage(pil_img)  # Armazena a img como atributo da instancia
        
        img_label = customtkinter.CTkLabel(img_frame, image=self.tk_img, text="")  # Remove o texto padrao
        img_label.pack(expand=True)

    def calcular_para_pasta(self):
        # Abre uma janela para o usuario selecionar o diretorio com as imagens.
        root = Tk()
        root.withdraw()  # Esconde a janela principal do Tkinter
        directory = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        
        # Se o usuario cancelar a selecao da pasta, interrompe a execucao.
        if not directory:
            print("Selecao de pasta cancelada pelo usuario.")
            return
        
        data = []
        
        # Loop em todos os arquivos no diretorio e calcula o SFM para a pasta.
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"): 
                filepath = os.path.join(directory, filename)
                
                # Carrega a img e considera a img inteira como ROI.
                f = io.imread(filepath, as_gray=True)
                features, labels = pyfeats.sfm_features(f, None, self.Lr, self.Lc)
                
                # Adiciona os resultados à lista de dados.
                row = [filename] + features.tolist()
                data.append(row)

        # Cria um DataFrame com os resultados utilizando pandas.
        columns = ['Filename'] + labels
        df = pd.DataFrame(data, columns=columns)
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="resultados_sfm_features.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        # Salva o DataFrame no arquivo CSV se o usuario escolher um caminho adequado.
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"CSV gerado com sucesso em: {save_path}")
        else:
            print("Salvamento cancelado pelo usuario.")

#=========================================
# Classificador SVM.
#=========================================

class SVMClassifier:
    def __init__(self,app, kernel='linear', C=2.0):
        self.app = app
        self.kernel = kernel
        self.C = C
        self.data = None
        self.conf_matrix_total = np.zeros((2, 2))
        self.unique_patients = None
        self.svm = None

        
        self.metrics_label = customtkinter.CTkLabel(
            app, text=":\nAcurácia: N/A\nSensibilidade: N/A\nEspecificidade: N/A"
        )
        self.metrics_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")        

    def load_data(self):
        # Carregar os dados do CSV
        file_path = filedialog.askopenfilename(title="Selecione o CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            raise ValueError("Nenhum arquivo CSV selecionado.")
        self.data = pd.read_csv(file_path, sep=";", encoding="latin1")

        # Mapear as classes para valores binários
        self.data['Classe'] = self.data['Classe'].map({'Esteatose Hepática': 1, 'Saudável': 0})

        # Extrair pacientes (assumindo que o ID do paciente esteja no nome da imagem)
        self.data['Paciente'] = self.data['Imagem'].str.extract(r'ROI_(\d+)', expand=False).astype(int)

        # Salvar pacientes únicos
        self.unique_patients = self.data['Paciente'].unique()

    @staticmethod
    def specificity_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def validate(self):
        X = self.data.drop(columns=['Imagem', 'Classe', 'Paciente'])
        y = self.data['Classe']

        for patient in self.unique_patients:
            # Dividir os dados entre treino e teste
            test_indices = self.data['Paciente'] == patient
            train_indices = ~test_indices

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            self.svm = SVC(kernel=self.kernel, C=self.C, class_weight='balanced')

            self.svm.fit(X_train, y_train)

            # Fazer previsões
            y_pred = self.svm.predict(X_test)

            # Atualizar a matriz de confusão total
            self.conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0, 1])

    def calculate_metrics(self):
        # Extrair valores da matriz de confusão total
        tn, fp, fn, tp = self.conf_matrix_total.ravel()

        # Cálculo das métricas
        accuracy = (tp + tn) / (tp + tn + fp + fn)  # Acurácia
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidade (Recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Especificidade
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precisão
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0  # F1-Score

        # Atualizar o Label ou exibir no terminal
        self.metrics_label.configure(
            text=(
                f"Métricas:\n"
                f"Acurácia Média: {accuracy:.2f}\n"
                f"Sensibilidade Média: {sensitivity:.2f}\n"
                f"Especificidade Média: {specificity:.2f}\n"
                f"F1-Score Médio: {f1_score:.2f}"
            )
        )
        print(self.metrics_label._text)
        self.app.update()

    def plot_confusion_matrix(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix_total.astype(int), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Saudável", "Esteatose Hepática"],
                    yticklabels=["Saudável", "Esteatose Hepática"])
        plt.title("Matriz de Confusão (Validação Cruzada)")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.show()

    def predict_image(self):
        """Permite que o usuário escolha uma imagem, a processa e realiza a predição."""
        # Abrir seletor de arquivo
        file_path = filedialog.askopenfilename(title="Selecione a imagem", 
                                            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            tkinter.messagebox.showinfo("Informação", "Nenhuma imagem selecionada.")
            return

        try:
            # Carregar imagem e extrair características SFM
            f = io.imread(file_path, as_gray=True)
            features, labels = pyfeats.sfm_features(f, None, 4, 4)
        except Exception as e:
            tkinter.messagebox.showerror("Erro", f"Erro ao carregar a imagem ou extrair características: {e}")
            return

        try:
            # Processar GLCM
            glcm = GLCMHandler.process_image(self=app, roi_path=file_path)
        except Exception as e:
            tkinter.messagebox.showerror("Erro", f"Erro ao processar GLCM: {e}")
            return

        # Combinar as características SFM e GLCM
        all_features = list(features) + list(glcm.values())
        
        # Criar o DataFrame no formato esperado pelo modelo
        columns = ['SFM_Coarseness', 'SFM_Contrast', 'SFM_Periodicity', 'SFM_Roughness',
                'entropy_d1', 'homogeneity_d1', 'entropy_d2', 'homogeneity_d2',
                'entropy_d4', 'homogeneity_d4', 'entropy_d8', 'homogeneity_d8']
        input_data = pd.DataFrame([all_features], columns=columns)

        # Verificar se o modelo foi treinado
        if self.svm is None:
            tkinter.messagebox.showwarning("Aviso", "O modelo SVM ainda não foi treinado.")
            return

        # Fazer a predição com o SVM treinado
        try:
            prediction = self.svm.predict(input_data)
            print(prediction)
            result = "Esteatose Hepática" if prediction[0] == 1 else "Saudável"
            tkinter.messagebox.showinfo("Resultado", f"A predição para a imagem é: {result}")
        except Exception as e:
            tkinter.messagebox.showerror("Erro", f"Erro ao realizar a predição: {e}")


    def show_metrics(self):
        self.metrics_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def hide_metrics(self):
        self.metrics_label.grid_remove()

#=========================================
# Classificador ResNet50.
#=========================================

class Resnet50:
    def __init__(self, app):
        self.app = app
        self.model = None
        # Caminho base das ROIs.
        self.base_path = None

    def select_folder(self):
        self.base_path = filedialog.askdirectory(
            title="Select Folder"
        )

    def preprocess_new_image(self, img_path, img_size=(224, 224)):
        img = load_img(img_path, target_size=img_size, color_mode="rgb")
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    # CLASSIFICACAO DE IMAGENS
    def load_model(self):
        file_path = filedialog.askopenfilename(title="Selecione o modelo", filetypes=[("H5 files", "*.h5")])
        if not file_path:
            raise ValueError("Nenhum modelo selecionado.")
        self.model = load_model(file_path)

    def classify_single_image(self):
        if not self.model:
            raise ValueError("Modelo não foi carregado. Por favor, carregue o modelo primeiro.")

        # Selecionar a imagem
        img_path = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("PNG files", "*.png")])
        if not img_path:
            raise ValueError("Nenhuma imagem selecionada.")

        # Preprocessar a imagem
        img_array = self.preprocess_new_image(img_path)

        # Fazer predição
        prediction = self.model.predict(img_array)
        predicted_class = (prediction > 0.5).astype(int)[0][0]

        # Determinar o diagnóstico
        diagnostico = 'Saudável' if predicted_class == 0 else 'Esteatose Hepática'

        # Atualizar métricas na interface
        self.metrics_label.configure(
            text=(f"Resultados:\n"
                  f"Imagem: {os.path.basename(img_path)}\n"
                  f"Diagnóstico: {diagnostico}")
        )

   
    # Funções Auxiliares.
   

    def load_and_preprocess_images(self, df, base_path, img_size=(224, 224)):
        X = []
        y = []
        for _, row in df.iterrows():
            img_path = os.path.join(base_path, row['Imagem'])
            img = load_img(img_path, target_size=img_size, color_mode="rgb")
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)  # Preprocessamento específico do ResNet50.
            X.append(img_array)
            y.append(row['Classe'])
        return np.array(X), np.array(y)

    def balance_data_with_augmentation(self, X, y):
        X_class0 = X[y == 0]
        y_class0 = y[y == 0]
        X_class1 = X[y == 1]
        y_class1 = y[y == 1]

        if len(y_class0) > len(y_class1):
            X_majority, y_majority = X_class0, y_class0
            X_minority, y_minority = X_class1, y_class1
        else:
            X_majority, y_majority = X_class1, y_class1
            X_minority, y_minority = X_class0, y_class0

        num_samples_to_generate = len(y_majority) - len(y_minority)

        # Gerador de aumento de dados para a classe minoritária.
        datagen_minority = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
        )

        X_minority_augmented = []
        y_minority_augmented = []

        iterations = math.ceil(num_samples_to_generate / len(X_minority))

        for i in range(iterations):
            for X_batch, y_batch in datagen_minority.flow(X_minority, y_minority, batch_size=len(X_minority), shuffle=False):
                X_minority_augmented.extend(X_batch)
                y_minority_augmented.extend(y_batch)
                if len(X_minority_augmented) >= num_samples_to_generate:
                    break
            if len(X_minority_augmented) >= num_samples_to_generate:
                break

        # Cortar o excesso de imagens geradas.
        X_minority_augmented = np.array(X_minority_augmented)[:num_samples_to_generate]
        y_minority_augmented = np.array(y_minority_augmented)[:num_samples_to_generate]

        X_balanced = np.vstack((X_majority, X_minority, X_minority_augmented))
        y_balanced = np.hstack((y_majority, y_minority, y_minority_augmented))

        indices = np.arange(len(y_balanced))
        np.random.shuffle(indices)
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]

        return X_balanced, y_balanced

    # modelo ResNet50 com regularização L2.
    def build_resnet50(self, input_shape=(224, 224, 3)):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # Congelar todas as camadas do modelo base.
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Regularização L2.
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model



    def run(self):
        # Configuração de Diretórios para Resultados.
        confusion_matrix_dir = "Matriz_Confusao_Rodada_ResNext"
        acc_graph_dir = "Grafico_Acuracia_Rodada_ResNext"
        results_dir = "Resultados_Finais_ResNext"

        os.makedirs(confusion_matrix_dir, exist_ok=True)
        os.makedirs(acc_graph_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
       
        # Carregamento de Dados.
        file_path = filedialog.askopenfilename(title="Selecione o CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            raise ValueError("Nenhum arquivo CSV selecionado.")

        try:
            data = pd.read_csv(file_path, sep=";", encoding="latin1")
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, sep=";", encoding="utf-8")
        except Exception as e:
            print(f"Erro ao ler o CSV: {e}")
            sys.exit(1)

        data.columns = data.columns.str.strip()

        print("Colunas do DataFrame:")
        print(data.columns.tolist())

        possible_class_column_names = ['Classe', 'classe', 'Class', 'class']

        for col in possible_class_column_names:
            if col in data.columns:
                class_column = col
                break
        else:
            raise KeyError(f"Nenhuma coluna correspondente a 'Classe' foi encontrada. As colunas disponíveis são: {data.columns.tolist()}")

        data[class_column] = data[class_column].map({'Esteatose Hepática': 1, 'Saudável': 0})

        data.rename(columns={class_column: 'Classe'}, inplace=True)

        if 'Imagem' not in data.columns:
            raise KeyError("A coluna 'Imagem' não foi encontrada no DataFrame. As colunas disponíveis são: {}".format(data.columns.tolist()))

        # Extrair o número do paciente.
        data['Paciente'] = data['Imagem'].str.extract(r'ROI_(\d+)', expand=False)

        # Verificar extração e converter para inteiro.
        if data['Paciente'].isnull().any():
            raise ValueError("Não foi possível extrair o número do paciente de algumas imagens. Verifique o padrão do nome dos arquivos.")

        data['Paciente'] = data['Paciente'].astype(int)

        print("Distribuição de classes total:")
        print(data['Classe'].value_counts())

        # Inicializar timer.
        start_time = time.time()

        # Loop de Treinamento por Paciente.
        accuracies = []
        conf_matrices = []

        history_accuracies = []
        history_val_accuracies = []
        history_losses = []
        history_val_losses = []

        unique_patients = data['Paciente'].unique()

        for patient in unique_patients:
            print(f"\nTreinando com paciente {patient} como teste...")

            test_data = data[data['Paciente'] == patient]
            train_data = data[data['Paciente'] != patient]

            X_train, y_train = self.load_and_preprocess_images(train_data, self.base_path)

            
            X_test, y_test = self.load_and_preprocess_images(test_data, self.base_path)

            print("Distribuição de classes no treinamento antes do balanceamento:")
            print(pd.Series(y_train).value_counts())

            # Aplicar balanceamento com aumento de dados.
            X_train_balanced, y_train_balanced = self.balance_data_with_augmentation(X_train, y_train)

            print("Distribuição de classes no treinamento após o balanceamento:")
            print(pd.Series(y_train_balanced).value_counts())

            # Aumento de dados para todo o conjunto de treinamento balanceado.
            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
            )

            X_train_augmented = []
            y_train_augmented = []

            for X_batch, y_batch in datagen.flow(X_train_balanced, y_train_balanced, batch_size=len(X_train_balanced), shuffle=False):
                X_train_augmented.extend(X_batch)
                y_train_augmented.extend(y_batch)
                if len(X_train_augmented) >= len(X_train_balanced) * 2:
                    break

            X_train_augmented = np.array(X_train_augmented)
            y_train_augmented = np.array(y_train_augmented)

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_augmented), y=y_train_augmented)
            class_weights = dict(enumerate(class_weights))

            model = self.build_resnet50(input_shape=(224, 224, 3))

            # Callbacks.
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            checkpoint = ModelCheckpoint('def.keras', save_best_only=True)

            # Treinar o modelo sem gerador.
            history = model.fit(
                X_train_augmented, y_train_augmented,
                validation_data=(X_test, y_test),
                epochs=5,
                callbacks=[early_stopping, checkpoint],
                verbose=1,
                class_weight=class_weights
            )

            model.save(f'modelo_paciente_{patient}.h5')

            # Armazenar gráficos de aprendizado.
            history_accuracies.append(history.history['accuracy'])
            history_val_accuracies.append(history.history['val_accuracy'])
            history_losses.append(history.history['loss'])
            history_val_losses.append(history.history['val_loss'])

            # Avaliação no conjunto de teste.
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            # Matrizes de confusão individuais.
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            conf_matrices.append(cm)

            unique_classes = np.unique(y_test)

            class_names = {0: 'Saudável', 1: 'Esteatose Hepática'}
            target_names = [class_names[cls] for cls in unique_classes]

            labels = [0, 1]
            target_names = ['Saudável', 'Esteatose Hepática']

            print("Relatório de Classificação:")
            print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

            # Plot matriz de confusao.
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Saudável", "Esteatose Hepática"],
                        yticklabels=["Saudável", "Esteatose Hepática"])
            plt.title(f"Matriz de Confusão - Paciente {patient}")
            plt.xlabel("Predito")
            plt.ylabel("Real")
            plt.savefig(os.path.join(confusion_matrix_dir, f"paciente_{patient}.png"))
            plt.close()

            # Plot gráfico de aprendizado.
            epochs_range = range(1, len(history.history['accuracy']) + 1)

            plt.figure(figsize=(12,5))

            # Gráfico de Acurácia.
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, history.history['accuracy'], label="Acurácia de Treino")
            plt.plot(epochs_range, history.history['val_accuracy'], label="Acurácia de Validação")
            plt.title(f"Acurácia por Época - Paciente {patient}")
            plt.xlabel("Épocas")
            plt.ylabel("Acurácia")
            plt.legend()

            # Gráfico de Loss.
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history.history['loss'], label="Loss de Treino")
            plt.plot(epochs_range, history.history['val_loss'], label="Loss de Validação")
            plt.title(f"Loss por Época - Paciente {patient}")
            plt.xlabel("Épocas")
            plt.ylabel("Loss")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(acc_graph_dir, f'paciente_{patient}.png'))
            plt.close()

        # Cálculo do tempo total
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTempo total para o treinamento: {elapsed_time:.2f} segundos.")

        # Resultados finais.
        mean_accuracy = np.mean(accuracies)
        print(f"\nMédia de Acurácia: {mean_accuracy:.4f}")

        conf_matrix_total = sum(conf_matrices)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_total.astype(int), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Saudável", "Esteatose Hepática"],
                    yticklabels=["Saudável", "Esteatose Hepática"])
        plt.title("Matriz de Confusão Total (Validação Cruzada)")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.savefig(os.path.join(results_dir, 'matriz_confusao_total.png'))
        plt.close()

        # Plotar gráficos de aprendizado médios.
        max_epochs = max([len(acc) for acc in history_accuracies])

        # Preencher com NaN para arrays de diferentes comprimentos.
        history_accuracies_padded = np.array([np.pad(acc, (0, max_epochs - len(acc)), 'constant', constant_values=np.nan) for acc in history_accuracies])
        history_val_accuracies_padded = np.array([np.pad(acc, (0, max_epochs - len(acc)), 'constant', constant_values=np.nan) for acc in history_val_accuracies])
        history_losses_padded = np.array([np.pad(loss, (0, max_epochs - len(loss)), 'constant', constant_values=np.nan) for loss in history_losses])
        history_val_losses_padded = np.array([np.pad(loss, (0, max_epochs - len(loss)), 'constant', constant_values=np.nan) for loss in history_val_losses])

        # Calcular a média ignorando os valores NaN.
        avg_accuracy = np.nanmean(history_accuracies_padded, axis=0)
        avg_val_accuracy = np.nanmean(history_val_accuracies_padded, axis=0)
        avg_loss = np.nanmean(history_losses_padded, axis=0)
        avg_val_loss = np.nanmean(history_val_losses_padded, axis=0)

        epochs_range = range(1, max_epochs + 1)

        plt.figure(figsize=(12, 5))

        # Gráfico de Acurácia.
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, avg_accuracy, label="Acurácia de Treino")
        plt.plot(epochs_range, avg_val_accuracy, label="Acurácia de Validação")
        plt.title("Acurácia Média por Época")
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.legend()

        # Gráfico de Loss.
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, avg_loss, label="Loss de Treino")
        plt.plot(epochs_range, avg_val_loss, label="Loss de Validação")
        plt.title("Loss Médio por Época")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'graficos_aprendizado.png'))
        plt.close()


if __name__ == "__main__":
    app = App()
    app.mainloop()