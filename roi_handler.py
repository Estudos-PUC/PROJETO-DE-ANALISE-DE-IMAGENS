import customtkinter
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import scipy.io
import os

class ROIHandler:
    def __init__(self, app):
        self.selected_patient_idx = None
        self.selected_img_idx = None
        self.img = None
        self.canvas = None
        self.rect = None
        self.click_x = None
        self.click_y = None
        self.save_button = None

    def recortar_roi(self):
        # Abrir nova janela para seleção de paciente e exibição de imagem
        self.recorte_window = customtkinter.CTkToplevel()
        self.recorte_window.title("Selecionar Paciente e Recortar ROI")
        self.recorte_window.geometry("1000x600")
        
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

        self.canvas = tkinter.Canvas(self.canvas_frame, width=600, height=600)
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

            self.img = Image.fromarray(image_data)

            self.tk_img = ImageTk.PhotoImage(self.img)

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

        # Tamanho do quadrado em pixels na imagem
        square_size = 28 

        # Coordenadas do retângulo na imagem
        x1 = event.x - square_size // 2
        y1 = event.y - square_size // 2
        x2 = event.x + square_size // 2
        y2 = event.y + square_size // 2

        # Desenhar retângulo verde
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=3)

        # Salvar as coordenadas do clique
        self.click_x = event.x
        self.click_y = event.y

        # Adicionar botão para salvar recorte, se ainda não existir
        if not self.save_button:
            self.save_button = customtkinter.CTkButton(self.recorte_window, text="Salvar Recorte", command=self.save_crop)
            self.save_button.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
            # Adicionar binding da tecla Enter para salvar recorte
            self.recorte_window.bind('<Return>', lambda event: self.save_crop())
    
    def save_crop(self):
        # Tamanho do quadrado em pixels na imagem original
        square_size = 28

        # Coordenadas do retângulo na imagem original
        x1 = self.click_x - (square_size // 2)
        y1 = self.click_y - (square_size // 2)
        x2 = self.click_x + (square_size // 2)
        y2 = self.click_y + (square_size // 2)

        # Garantir que as coordenadas estejam dentro dos limites da imagem
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.img.width, x2)
        y2 = min(self.img.height, y2)

        # Recortar a imagem
        crop_box = (x1, y1, x2, y2)
        cropped_img = self.img.crop(crop_box)

        # Criar o nome de arquivo padrão
        filename = f"ROI_{self.selected_patient_idx:02d}_{self.selected_img_idx}"

        # Abrir o diálogo "Salvar Como" com o nome de arquivo padrão
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=filename,
            filetypes=[("PNG files", "*.png")]
        )

        if save_path:
            # Salvar a imagem recortada
            cropped_img.save(save_path)

            # Salvar as coordenadas (x1, y1) em um arquivo de texto compartilhado
            coord_save_path = "ROI_coordinates.txt"
            with open(coord_save_path, 'a') as f:
                f.write(f"{filename}: Coordinates of top-left corner: (x={x1}, y={y1})\n")

            tkinter.messagebox.showinfo("Salvo", f"Imagem recortada salva em {save_path}\nCoordenadas salvas em {coord_save_path}")

    def calcular_hi_e_ajustar_figado(self):
        # Diretórios das ROIs
        figado_dir = "./figado"
        rim_dir = "./rim"
        output_dir = "./Figado_Ajustado"
        os.makedirs(output_dir, exist_ok=True)

        # Arquivo para salvar os valores de HI
        hi_save_path = "HI_values.txt"

        # Iterar sobre todas as ROIs do fígado
        with open(hi_save_path, 'w') as hi_file:
            for filename in os.listdir(figado_dir):
                if filename.endswith(".png") and "RIM" not in filename:
                    # Carregar a ROI do fígado e do rim correspondente
                    figado_path = os.path.join(figado_dir, filename)
                    rim_path = os.path.join(rim_dir, filename.replace('.png', '_RIM.png'))

                    if not os.path.exists(rim_path):
                        continue

                    figado_img = Image.open(figado_path).convert("L")
                    rim_img = Image.open(rim_path).convert("L")

                    # Converter as imagens para arrays NumPy
                    figado_array = np.array(figado_img)
                    rim_array = np.array(rim_img)

                    # Calcular a média dos tons de cinza
                    media_figado = np.mean(figado_array)
                    media_rim = np.mean(rim_array)

                    # Calcular o índice hepatorenal (HI)
                    hi = media_figado / media_rim if media_rim != 0 else 1

                    # Salvar o valor de HI no arquivo
                    hi_file.write(f"{filename.replace('.png', '')}, {hi}\n")

                    # Ajustar os tons de cinza da ROI do fígado
                    ajustado_figado_array = np.round(figado_array * hi).astype(np.uint8)

                    # Criar imagem ajustada do fígado
                    ajustado_figado_img = Image.fromarray(ajustado_figado_array)

                    # Salvar a ROI ajustada do fígado
                    ajustado_figado_path = os.path.join(output_dir, filename)
                    ajustado_figado_img.save(ajustado_figado_path)
        
        # Mostrar todas as mensagens de sucesso no final
        tkinter.messagebox.showinfo("Salvo", f"Imagens ajustadas e salvas em {output_dir}\n valores do HI salvos em {hi_save_path}")
