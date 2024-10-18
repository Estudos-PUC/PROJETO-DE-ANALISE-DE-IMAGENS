import customtkinter
from app_config import AppConfig
from image_handler import ImageHandler
from roi_handler import ROIHandler
from glcm_handler import GLCMHandler

# Parte 4: Classe Principal do App
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.config = AppConfig(self)
        self.image_handler = ImageHandler(self)
        self.roi_handler = ROIHandler(self)
        self.glcm_handler = GLCMHandler(self)

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
        print("Botão da barra lateral clicado")

    def calcular_hi_imagem(self):
        self.roi_handler.calcular_hi_imagem()
        
            
if __name__ == "__main__":
    app = App()
    app.mainloop()