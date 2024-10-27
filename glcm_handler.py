import numpy as np
from PIL import Image, ImageTk
import tkinter.messagebox
import tkinter
import customtkinter # Para GUI
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops # Para GLCM
from skimage.util import img_as_ubyte
import os
import tkinter.ttk as ttk
import pandas as pd  # Para operações de CSV
import re # Para RegEx

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

    # Calcula a GLCM de uma imagem de acordo com os ângulos possíveis (de 0 a 360),
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
