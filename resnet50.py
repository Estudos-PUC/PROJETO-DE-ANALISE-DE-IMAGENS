import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import math

# Diretorio das imgs
confusion_matrix_dir = "Matriz_Confusao_Rodada_ResNext"
acc_graph_dir = "Grafico_Acuracia_Rodada_ResNext"
results_dir = "Resultados_Finais_ResNext"

os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(acc_graph_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def preprocess_new_image(img_path, img_size=(224, 224)):
    img = load_img(img_path, target_size=img_size, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão batch.
    return img_array

# Caminho base das ROIs.
base_path = r"/Users/edu/Documents/PROJETO-DE-ANALISE-DE-IMAGENS/Figado_Ajustado"

def load_and_preprocess_images(df, base_path, img_size=(224, 224)):
    X = []
    y = []
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['Imagem'])
        img = load_img(img_path, target_size=img_size, color_mode="rgb")
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)  # Preprocessamento específico do ResNet50.
        X.append(img_array)
        y.append(row['Classe'])
    return np.array(X), np.array(y)

def balance_data_with_augmentation(X, y):
    X_class0 = X[y == 0]
    y_class0 = y[y == 0]
    X_class1 = X[y == 1]
    y_class1 = y[y == 1]

    if len(y_class0) > len(y_class1):
        X_majority, y_majority = X_class0, y_class0
        X_minority, y_minority = X_class1, y_class1
    else:
        X_majority, y_majority = X_class1, y_class1
        X_minority, y_minority = X_class0, y_class0

    num_samples_to_generate = len(y_majority) - len(y_minority)

    # Gerador de aumento de dados para a classe minoritária.
    datagen_minority = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )

    X_minority_augmented = []
    y_minority_augmented = []

    iterations = math.ceil(num_samples_to_generate / len(X_minority))

    for i in range(iterations):
        for X_batch, y_batch in datagen_minority.flow(X_minority, y_minority, batch_size=len(X_minority), shuffle=False):
            X_minority_augmented.extend(X_batch)
            y_minority_augmented.extend(y_batch)
            if len(X_minority_augmented) >= num_samples_to_generate:
                break
        if len(X_minority_augmented) >= num_samples_to_generate:
            break

    # Cortar o excesso de imagens geradas.
    X_minority_augmented = np.array(X_minority_augmented)[:num_samples_to_generate]
    y_minority_augmented = np.array(y_minority_augmented)[:num_samples_to_generate]

    X_balanced = np.vstack((X_majority, X_minority, X_minority_augmented))
    y_balanced = np.hstack((y_majority, y_minority, y_minority_augmented))

    indices = np.arange(len(y_balanced))
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]

    return X_balanced, y_balanced

# modelo ResNet50 com regularização L2.
def build_resnet50(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Congelar todas as camadas do modelo base.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Regularização L2.
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Configuração principal.
file_path = filedialog.askopenfilename(title="Selecione o CSV", filetypes=[("CSV files", "*.csv")])
if not file_path:
    raise ValueError("Nenhum arquivo CSV selecionado.")

try:
    data = pd.read_csv(file_path, sep=";", encoding="latin1")
except UnicodeDecodeError:
    data = pd.read_csv(file_path, sep=";", encoding="utf-8")
except Exception as e:
    print(f"Erro ao ler o CSV: {e}")
    sys.exit(1)

data.columns = data.columns.str.strip()

print("Colunas do DataFrame:")
print(data.columns.tolist())

possible_class_column_names = ['Classe', 'classe', 'Class', 'class']

for col in possible_class_column_names:
    if col in data.columns:
        class_column = col
        break
else:
    raise KeyError(f"Nenhuma coluna correspondente a 'Classe' foi encontrada. As colunas disponíveis são: {data.columns.tolist()}")

data[class_column] = data[class_column].map({'Esteatose Hepática': 1, 'Saudável': 0})

data.rename(columns={class_column: 'Classe'}, inplace=True)

if 'Imagem' not in data.columns:
    raise KeyError("A coluna 'Imagem' não foi encontrada no DataFrame. As colunas disponíveis são: {}".format(data.columns.tolist()))

# Extrair o número do paciente.
data['Paciente'] = data['Imagem'].str.extract(r'ROI_(\d+)', expand=False)

# Verificar extração e converter para inteiro.
if data['Paciente'].isnull().any():
    raise ValueError("Não foi possível extrair o número do paciente de algumas imagens. Verifique o padrão do nome dos arquivos.")

data['Paciente'] = data['Paciente'].astype(int)

print("Distribuição de classes total:")
print(data['Classe'].value_counts())

accuracies = []
conf_matrices = []

history_accuracies = []
history_val_accuracies = []
history_losses = []
history_val_losses = []

unique_patients = data['Paciente'].unique()

for patient in unique_patients:
    print(f"\nTreinando com paciente {patient} como teste...")

    test_data = data[data['Paciente'] == patient]
    train_data = data[data['Paciente'] != patient]

    X_train, y_train = load_and_preprocess_images(train_data, base_path)
    X_test, y_test = load_and_preprocess_images(test_data, base_path)

    print("Distribuição de classes no treinamento antes do balanceamento:")
    print(pd.Series(y_train).value_counts())

    # Aplicar balanceamento com aumento de dados.
    X_train_balanced, y_train_balanced = balance_data_with_augmentation(X_train, y_train)

    print("Distribuição de classes no treinamento após o balanceamento:")
    print(pd.Series(y_train_balanced).value_counts())

    # Aumento de dados para todo o conjunto de treinamento balanceado.
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
    )

    X_train_augmented = []
    y_train_augmented = []

    for X_batch, y_batch in datagen.flow(X_train_balanced, y_train_balanced, batch_size=len(X_train_balanced), shuffle=False):
        X_train_augmented.extend(X_batch)
        y_train_augmented.extend(y_batch)
        if len(X_train_augmented) >= len(X_train_balanced) * 2:
            break

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_augmented), y=y_train_augmented)
    class_weights = dict(enumerate(class_weights))

    model = build_resnet50(input_shape=(224, 224, 3))

    # Callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('def.keras', save_best_only=True)

    # Treinar o modelo sem gerador.
    history = model.fit(
        X_train_augmented, y_train_augmented,
        validation_data=(X_test, y_test),
        epochs=5,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        class_weight=class_weights
    )

    model.save(f'modelo_paciente_{patient}.h5')

    # Armazenar gráficos de aprendizado.
    history_accuracies.append(history.history['accuracy'])
    history_val_accuracies.append(history.history['val_accuracy'])
    history_losses.append(history.history['loss'])
    history_val_losses.append(history.history['val_loss'])

    # Avaliação no conjunto de teste.
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Matrizes de confusão individuais.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    conf_matrices.append(cm)

    unique_classes = np.unique(y_test)

    class_names = {0: 'Saudável', 1: 'Esteatose Hepática'}
    target_names = [class_names[cls] for cls in unique_classes]

    labels = [0, 1]
    target_names = ['Saudável', 'Esteatose Hepática']

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # Plot matriz de confusao.
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Saudável", "Esteatose Hepática"],
                yticklabels=["Saudável", "Esteatose Hepática"])
    plt.title(f"Matriz de Confusão - Paciente {patient}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(os.path.join(confusion_matrix_dir, f"paciente_{patient}.png"))
    plt.close()

    # Plot gráfico de aprendizado.
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12,5))

    # Gráfico de Acurácia.
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label="Acurácia de Treino")
    plt.plot(epochs_range, history.history['val_accuracy'], label="Acurácia de Validação")
    plt.title(f"Acurácia por Época - Paciente {patient}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()

    # Gráfico de Loss.
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label="Loss de Treino")
    plt.plot(epochs_range, history.history['val_loss'], label="Loss de Validação")
    plt.title(f"Loss por Época - Paciente {patient}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(acc_graph_dir, f'paciente_{patient}.png'))
    plt.close()

#=========================================
# Resultados finais
#=========================================
mean_accuracy = np.mean(accuracies)
print(f"\nMédia de Acurácia: {mean_accuracy:.4f}")

conf_matrix_total = sum(conf_matrices)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_total.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Saudável", "Esteatose Hepática"],
            yticklabels=["Saudável", "Esteatose Hepática"])
plt.title("Matriz de Confusão Total (Validação Cruzada)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig(os.path.join(results_dir, 'matriz_confusao_total.png'))
plt.close()

# Plotar gráficos de aprendizado médios.
max_epochs = max([len(acc) for acc in history_accuracies])

# Preencher com NaN para arrays de diferentes comprimentos.
history_accuracies_padded = np.array([np.pad(acc, (0, max_epochs - len(acc)), 'constant', constant_values=np.nan) for acc in history_accuracies])
history_val_accuracies_padded = np.array([np.pad(acc, (0, max_epochs - len(acc)), 'constant', constant_values=np.nan) for acc in history_val_accuracies])
history_losses_padded = np.array([np.pad(loss, (0, max_epochs - len(loss)), 'constant', constant_values=np.nan) for loss in history_losses])
history_val_losses_padded = np.array([np.pad(loss, (0, max_epochs - len(loss)), 'constant', constant_values=np.nan) for loss in history_val_losses])

# Calcular a média ignorando os valores NaN.
avg_accuracy = np.nanmean(history_accuracies_padded, axis=0)
avg_val_accuracy = np.nanmean(history_val_accuracies_padded, axis=0)
avg_loss = np.nanmean(history_losses_padded, axis=0)
avg_val_loss = np.nanmean(history_val_losses_padded, axis=0)

epochs_range = range(1, max_epochs + 1)

plt.figure(figsize=(12, 5))

# Gráfico de Acurácia.
plt.subplot(1, 2, 1)
plt.plot(epochs_range, avg_accuracy, label="Acurácia de Treino")
plt.plot(epochs_range, avg_val_accuracy, label="Acurácia de Validação")
plt.title("Acurácia Média por Época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

# Gráfico de Loss.
plt.subplot(1, 2, 2)
plt.plot(epochs_range, avg_loss, label="Loss de Treino")
plt.plot(epochs_range, avg_val_loss, label="Loss de Validação")
plt.title("Loss Médio por Época")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'graficos_aprendizado.png'))
plt.close()
