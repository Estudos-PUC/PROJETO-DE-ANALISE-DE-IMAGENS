import customtkinter

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
        self.sidebar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
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

        # Botao lateral Recortar ROI
        self.recortar_roi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Recortar ROI",
            command=app.recortar_roi,
            width=200
        )
        self.recortar_roi_button.grid(row=2, column=0, padx=20, pady=10)

        # Botao lateral Calcular HI
        self.calcular_hi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Calcular HI",
            command=app.calcular_hi_imagem,
            width=200
        )
        self.calcular_hi_button.grid(row=3, column=0, padx=20, pady=10)

        # Botao lateral Visualizar Histograma
        self.visualizar_histograma_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Visualizar Histograma",
            command=app.gerar_histograma,
            width=200
        )
        self.visualizar_histograma_button.grid(row=4, column=0, padx=20, pady=10)

        # Botao lateral Computar GLCM
        self.computar_glcm_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Computar GLCM",
            command=app.calcular_glcm,
            width=200
        )
        self.computar_glcm_button.grid(row=5, column=0, padx=20, pady=10)

        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        # Botao lateral Caracterizar ROI
        self.caracterizar_roi_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Caracterizar ROI",
            command=app.sidebar_button_event,
            width=200
        )
        self.caracterizar_roi_button.grid(row=6, column=0, padx=20, pady=10)

        # Botao lateral Classificar Imagem
        self.classificar_imagem_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Classificar Imagem",
            command=app.sidebar_button_event,
            width=200
        )
        self.classificar_imagem_button.grid(row=7, column=0, padx=20, pady=10)
