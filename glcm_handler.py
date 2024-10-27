# glcm_handler.py

import os
import numpy as np
from PIL import Image, ImageTk
import tkinter.messagebox
import tkinter
import customtkinter
from tkinter import filedialog
import pyfeats
import re
import pandas as pd

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
            roi_array = roi_array.astype(np.uint8)  # Ensure the image is in uint8 format

            distances = [1, 2, 4, 8]
            angles = np.deg2rad(np.arange(0, 360, 1))  # Angles from 0° to 359°

            # Compute GLCM features using pyfeats
            features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(
                roi_array,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normalized=True,
                ignore_zeros=True
            )

            # Combine features and labels into dictionaries
            features_mean_dict = dict(zip(labels_mean, features_mean))
            features_range_dict = dict(zip(labels_range, features_range))

            # Prepare the features to display (you can choose which ones to show)
            features_to_display = {}
            for label in labels_mean:
                mean_value = features_mean_dict[label]
                range_value = features_range_dict[label.replace('Mean', 'Range')]
                features_to_display[f"{label}"] = mean_value
                features_to_display[f"{label.replace('Mean', 'Range')}"] = range_value

            self.display_features(roi_path, features_to_display)

    def display_features(self, roi_path, features):
        # Create a new window to display the features
        feature_window = customtkinter.CTkToplevel(self.app)
        feature_window.title("Descritores de Textura - GLCM")
        feature_window.geometry("600x600")  # Increased height for more features

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

    def calcular_glcm_todas_rois(self):
        # Prompt the user to select the directory containing the ROIs
        roi_dir = filedialog.askdirectory(title="Selecione o diretório das ROIs")
        if not roi_dir:
            return  # User cancelled the selection

        output_csv = "glcm_all_features.csv"

        # Prepare lists to store features and labels
        features_list = []

        # Obtain and sort filenames
        filenames = [f for f in os.listdir(roi_dir) if f.endswith('.png')]

        # Function to extract numbers from filenames
        def extract_numbers(filename):
            # Pattern to extract numbers: ROI_XX_YY.png
            match = re.match(r'ROI_(\d+)_(\d+)\.png', filename)
            if match:
                patient_num = int(match.group(1))
                image_num = int(match.group(2))
                return (patient_num, image_num)
            else:
                # If filename doesn't match the pattern, place it at the end
                return (float('inf'), float('inf'))

        # Sort the filenames
        filenames_sorted = sorted(filenames, key=extract_numbers)

        # Loop over the sorted filenames
        for filename in filenames_sorted:
            roi_path = os.path.join(roi_dir, filename)

            # Load the ROI image
            roi_img = Image.open(roi_path).convert("L")
            roi_array = np.array(roi_img).astype(np.uint8)

            distances = [1, 2, 4, 8]
            angles = np.deg2rad(np.arange(0, 360, 1))  # Angles from 0° to 359°

            # Compute GLCM features using pyfeats
            features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(
                roi_array,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normalized=True,
                ignore_zeros=True
            )

            # Combine features and labels into dictionaries
            features_mean_dict = dict(zip(labels_mean, features_mean))
            features_range_dict = dict(zip(labels_range, features_range))

            # Combine mean and range features
            all_features = {}
            for label in labels_mean:
                mean_value = features_mean_dict[label]
                range_value = features_range_dict[label.replace('Mean', 'Range')]
                all_features[f"{label}"] = mean_value
                all_features[f"{label.replace('Mean', 'Range')}"] = range_value

            # Add filename and patient/image indices
            patient_num, image_num = extract_numbers(filename)
            all_features['filename'] = filename
            all_features['patient_num'] = patient_num
            all_features['image_num'] = image_num

            # Append to the features list
            features_list.append(all_features)

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(features_list)
        df.to_csv(output_csv, index=False)

        tkinter.messagebox.showinfo("Concluído", f"Características GLCM calculadas e salvas em {output_csv}")
