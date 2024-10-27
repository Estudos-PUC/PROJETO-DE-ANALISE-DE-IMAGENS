import customtkinter
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import scipy.io
import os

SQUARE_SIZE = 28
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
                tkinter.messagebox.showerror("Erro", "Os dados da imagem nao sao validos.")
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
                tkinter.messagebox.showerror("Erro", "Os dados da imagem nao sao validos.")
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
