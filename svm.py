from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados do CSV
file_path = filedialog.askopenfilename(title="Selecione o CSV", filetypes=[("CSV files", "*.csv")])
if not file_path:
    raise ValueError("Nenhum arquivo CSV selecionado.")

# Ler os dados
data = pd.read_csv(file_path, sep=";", encoding="latin1")

# Mapear as classes para valores binários
data['Classe'] = data['Classe'].map({'Esteatose Hepática': 1, 'Saudável': 0})

# Extrair pacientes (assumindo que o ID do paciente esteja no nome da imagem)
data['Paciente'] = data['Imagem'].str.extract(r'ROI_(\d+)', expand=False).astype(int)

# Separar características e classes
X = data.drop(columns=['Imagem', 'Classe', 'Paciente'])
y = data['Classe']

# Função para calcular especificidade
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# Listas para armazenar métricas
accuracies = []
sensitivities = []
specificities = []
f1_scores = []
conf_matrix_total = np.zeros((2, 2))

# Validação cruzada leave-one-patient-out
unique_patients = data['Paciente'].unique()

for patient in unique_patients:
    # Dividir os dados entre treino e teste
    test_indices = data['Paciente'] == patient
    train_indices = ~test_indices

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Treinar o classificador SVM
    svm = SVC(kernel='linear', C=2.0)  # Considera o desequilíbrio no treino
    svm.fit(X_train, y_train)

    # Fazer previsões
    y_pred = svm.predict(X_test)

    # Calcular métricas
    accuracies.append(accuracy_score(y_test, y_pred))
    sensitivities.append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))  # Sensibilidade
    specificities.append(specificity_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))  # F1-Score

    # Atualizar a matriz de confusão total
    conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0, 1])

# Calcular métricas médias
mean_accuracy = np.mean(accuracies)
mean_sensitivity = np.mean(sensitivities)
mean_specificity = np.mean(specificities)
mean_f1_score = np.mean(f1_scores)

# Exibir resultados
print(f"Média de Acurácia: {mean_accuracy:.4f}")
print(f"Média de Sensibilidade: {mean_sensitivity:.4f}")
print(f"Média de Especificidade: {mean_specificity:.4f}")
print(f"Média de F1-Score: {mean_f1_score:.4f}")

# Mostrar a matriz de confusão total
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_total.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Saudável", "Esteatose Hepática"],
            yticklabels=["Saudável", "Esteatose Hepática"])
plt.title("Matriz de Confusão (Validação Cruzada)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
