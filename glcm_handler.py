import os
import numpy as np
from PIL import Image, ImageTk
import tkinter.messagebox
import tkinter
import customtkinter
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops

class GLCMHandler:
    def __init__(self, app):
        self.app = app

    def computar_glcm_roi(self):
        roi_path = filedialog.askopenfilename(
            title="Selecione uma ROI",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if roi_path:
            roi_img = Image.open(roi_path).convert("L")
            roi_array = np.array(roi_img)

            distances = [1, 2, 4, 8]
            angles = [0] 

            glcm = graycomatrix(
                roi_array,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True
            )

            features = {}
            for i, distance in enumerate(distances):
                homogeneity = graycoprops(glcm, 'homogeneity')[i, 0]

                # Compute Entropy
                glcm_matrix = glcm[:, :, i, 0]
                glcm_prob = glcm_matrix / np.sum(glcm_matrix)
                glcm_prob_nonzero = glcm_prob + (glcm_prob == 0) * 1e-10
                entropy = -np.sum(glcm_prob_nonzero * np.log(glcm_prob_nonzero))

                features[f'Homogeneidade (d={distance})'] = homogeneity
                features[f'Entropia (d={distance})'] = entropy

            self.display_features(roi_path, features)

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