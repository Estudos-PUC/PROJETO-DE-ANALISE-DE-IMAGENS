import numpy as np
from PIL import Image, ImageTk
import tkinter.messagebox
import tkinter
import customtkinter
from tkinter import filedialog
from skimage.feature import graycomatrix
from skimage.util import img_as_ubyte
from itertools import product
import os
import tkinter.ttk as ttk
import pandas as pd  # Added pandas for CSV operations

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
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
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

            # Save to CSV
            self.save_features_to_csv(all_features)

    def process_image(self, roi_path):
        roi_img = Image.open(roi_path).convert("L")
        roi_array = np.array(roi_img)
        roi_array = img_as_ubyte(roi_array)  # Ensure image is in uint8 format

        distances = [1, 2, 4, 8]

        features = {}
        for distance in distances:
            # Generate all possible offsets for the given distance
            offsets = self.generate_offsets(distance)
            # Initialize GLCM sum
            glcm_sum = np.zeros((256, 256), dtype=np.float64)
            for offset in offsets:
                # Compute GLCM for each offset
                glcm = graycomatrix(
                    roi_array,
                    distances=[distance],
                    angles=[np.arctan2(offset[1], offset[0])],
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                glcm_sum += glcm[:, :, 0, 0]  # Sum GLCMs

            # Normalize the aggregated GLCM
            glcm_sum /= glcm_sum.sum()

            # Compute Homogeneity
            i, j = np.indices(glcm_sum.shape)
            homogeneity = np.sum(glcm_sum / (1 + np.abs(i - j)))

            # Compute Entropy
            glcm_prob_nonzero = glcm_sum + (glcm_sum == 0) * 1e-10
            entropy = -np.sum(glcm_prob_nonzero * np.log(glcm_prob_nonzero))

            # Store features with the desired keys
            features[f'entropy_d{distance}'] = entropy
            features[f'homogeneity_d{distance}'] = homogeneity

        return features

    def generate_offsets(self, distance):
        # Generate all integer offsets within the circle of given radius
        offsets = []
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if dx == 0 and dy == 0:
                    continue
                if round(np.hypot(dx, dy)) == distance:
                    offsets.append((dx, dy))
        # Remove duplicates
        offsets = list(set(offsets))
        return offsets

    def display_features(self, roi_path, features):
        # Create a new window to display the features
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM")
        feature_window.geometry("600x400")

        # Display the ROI image
        img_frame = customtkinter.CTkFrame(feature_window)
        img_frame.grid(row=0, column=0, padx=10, pady=10)

        img_label = customtkinter.CTkLabel(img_frame, text="ROI Selecionada")
        img_label.pack()

        roi_img = Image.open(roi_path).resize((200, 200))
        roi_photo = ImageTk.PhotoImage(roi_img)
        img_canvas = tkinter.Canvas(img_frame, width=200, height=200)
        img_canvas.pack()
        img_canvas.create_image(0, 0, anchor="nw", image=roi_photo)
        img_canvas.image = roi_photo  # Keep a reference

        # Display the features
        feature_frame = customtkinter.CTkFrame(feature_window)
        feature_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        feature_label = customtkinter.CTkLabel(feature_frame, text="Valores Calculados", font=("Arial", 16))
        feature_label.pack(pady=(0, 10))

        # Create a listbox to display the features
        listbox = tkinter.Listbox(feature_frame, font=("Arial", 12))
        listbox.pack(fill="both", expand=True)

        for key, value in features.items():
            listbox.insert(tkinter.END, f"{key}: {value:.4f}")

        # Adjust window grid configuration
        feature_window.grid_rowconfigure(0, weight=1)
        feature_window.grid_columnconfigure(1, weight=1)

    def display_features_directory(self, all_features):
        # Create a new window to display the features
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM (Diretório)")
        feature_window.geometry("800x600")

        # Create a frame to hold the table
        table_frame = customtkinter.CTkFrame(feature_window)
        table_frame.pack(fill="both", expand=True)

        # Use tkinter Treeview to display data in tabular format
        columns = ['Imagem', 'entropy_d1', 'homogeneity_d1', 'entropy_d2', 'homogeneity_d2',
                   'entropy_d4', 'homogeneity_d4', 'entropy_d8', 'homogeneity_d8']
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        tree.pack(fill="both", expand=True)

        # Define headings
        for col in columns:
            tree.heading(col, text=col)

        # Insert data
        for features in all_features:
            row = [features.get(col, '') for col in columns]
            tree.insert('', tkinter.END, values=row)

        # Add scrollbar
        scrollbar = tkinter.Scrollbar(tree, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    def save_features_to_csv(self, all_features):
        # Convert the list of feature dictionaries into a DataFrame
        df = pd.DataFrame(all_features)

        # Ensure columns are in the desired order
        columns = ['Imagem', 'entropy_d1', 'homogeneity_d1', 'entropy_d2', 'homogeneity_d2',
                   'entropy_d4', 'homogeneity_d4', 'entropy_d8', 'homogeneity_d8']
        df = df[columns]

        # Ask user where to save the CSV file
        csv_path = filedialog.asksaveasfilename(
            title="Salvar arquivo CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if csv_path:
            # Save the DataFrame to CSV
            df.to_csv(csv_path, index=False)
            tkinter.messagebox.showinfo("Sucesso", f"Arquivo CSV salvo em: {csv_path}")
