import os
import pandas as pd
import pyfeats
from PIL import Image, ImageTk
from skimage import io
from tkinter import Tk, filedialog
import customtkinter as ctk

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
        
        feature_window = ctk.CTkToplevel(self.app)
        feature_window.title("Resultados das Características SFM")
        feature_window.geometry("1000x700")

        table_frame = ctk.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True, side="left", padx=10, pady=10)

        text_widget = ctk.CTkTextbox(table_frame, wrap="word")
        text_widget.pack(fill="both", expand=True)

        text_widget.insert("end", f"Nome do Arquivo: {os.path.basename(filepath)}\n\n")
        text_widget.insert("end", "Características:\n")
        
        for label, feature in zip(labels, features):
            text_widget.insert("end", f"{label}: {feature}\n")
        
        text_widget.configure(state="disabled")
        
        img_frame = ctk.CTkFrame(feature_window, width=500, height=500)
        img_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        pil_img = Image.open(filepath)
        pil_img = pil_img.resize((500, 500), Image.LANCZOS)  # Redimensiona a imagem para 500x500
        self.tk_img = ImageTk.PhotoImage(pil_img)  # Armazena a imagem como atributo da instância
        
        img_label = ctk.CTkLabel(img_frame, image=self.tk_img, text="")  # Remove o texto padrão
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
            if filename.endswith(".png") or filename.endswith(".jpg"):  # Filtra para arquivos de imagem
                filepath = os.path.join(directory, filename)
                
                # Carrega a imagem e considera a imagem inteira como ROI
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


