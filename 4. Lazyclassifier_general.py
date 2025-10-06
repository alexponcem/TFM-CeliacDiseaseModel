from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd

# DATASET CON DATOS CLÍNICOS
# Dataset de diferencias con datos clínicos
#ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_diferencias_DC.csv"
# Dataset concatenado con datos clínicos
#ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_concatenado_DC.csv"

# Dataset de diferencias con datos clínicos filtrados
#ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_diferencias_DC_fil.csv"
# Dataset concatenado con datos clínicos filtrados
ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv"

final_df = pd.read_csv(ruta)

# Convertir la columna "Etiqueta" a numérica e imprimir conteo
final_df["Etiqueta"] = final_df["Etiqueta"].map({"Negativo": 0, "Positivo": 1})
print(final_df["Etiqueta"].value_counts())

# Cargar datos (asegurarse de que 'final_df' ya esté definido)
X = final_df.drop(["Etiqueta", "Patient_code"], axis=1)
y = final_df["Etiqueta"]

# Umbral de varianza (ajustado según necesidad: 0.01 para eliminar casi constantes)
threshold = 0.01  
selector = VarianceThreshold(threshold=threshold)
X_reduced = selector.fit_transform(X)

# Obtener nombres de las columnas seleccionadas
selected_features = X.columns[selector.get_support()]
print(f"Número de columnas originales: {X.shape[1]}")
print(f"Número de columnas después de filtrar: {X_reduced.shape[1]}")
print(f"Columnas eliminadas: {set(X.columns) - set(selected_features)}")

# Dividir datos en train y test (antes de escalar para evitar data leakage)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Escalar solo las features numéricas
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a DataFrame (opcional, para mejor visualización)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

# Estadísticas post-escalado
print("\nEstadísticas después del escalado (Train):")
print(pd.DataFrame(X_train_scaled).describe().loc[['mean', 'std', 'min', 'max']])

# Ejemplo de visualización de una columna
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(X_train[:, 0], bins=30, color='blue', alpha=0.7)
plt.title("Antes de escalar")
plt.subplot(1, 2, 2)
plt.hist(X_train_scaled.iloc[:, 0], bins=30, color='red', alpha=0.7)
plt.title("Después de escalar")
plt.show()

model = LGBMClassifier(min_data_in_leaf=1, verbosity=-1)
model.fit(X_train_scaled, y_train)

# Entrenar y comparar modelos
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)
# Mostrar resultados
print(models)

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from lazypredict.Supervised import LazyClassifier
# from lightgbm import LGBMClassifier
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import RobustScaler

# # Lista de rutas de tus datasets
# rutas = [
#     r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_diferencias_DC.csv",
#     r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_concatenado_DC.csv",
#     r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_diferencias_DC_fil.csv",
#     r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv"
# ]

# # Parámetro para activar/desactivar el escalado
# usar_escalado = False  # Cambia a False si NO quieres escalar

# resultados = []

# for ruta in rutas:
#     print(f"\nAnalizando: {os.path.basename(ruta)}")
#     final_df = pd.read_csv(ruta)

#     # Convertir la columna "Etiqueta" a numérica e imprimir conteo
#     final_df["Etiqueta"] = final_df["Etiqueta"].map({"Negativo": 0, "Positivo": 1})
#     print(final_df["Etiqueta"].value_counts())
          
#     # Cargar datos (asegurarse de que 'final_df' ya esté definido)
#     X = final_df.drop(["Etiqueta", "Patient_code"], axis=1)
#     y = final_df["Etiqueta"]

#     # Filtrado por varianza
#     threshold = 0.01
#     selector = VarianceThreshold(threshold=threshold)
#     X_reduced = selector.fit_transform(X)

#     # Obtener nombres de las columnas seleccionadas
#     selected_features = X.columns[selector.get_support()]
#     print(f"Número de columnas originales: {X.shape[1]}")
#     print(f"Número de columnas después de filtrar: {X_reduced.shape[1]}")
#     print(f"Columnas eliminadas: {set(X.columns) - set(selected_features)}")

#     # Dividir datos en train y test (antes de escalar para evitar data leakage)
#     X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

#     # Escalar solo las features numéricas
#     if usar_escalado:
#         scaler = RobustScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         X_train_final = pd.DataFrame(X_train_scaled, columns=selected_features)
#         X_test_final = pd.DataFrame(X_test_scaled, columns=selected_features)
#     else:
#         X_train_final = pd.DataFrame(X_train, columns=selected_features)
#         X_test_final = pd.DataFrame(X_test, columns=selected_features)

#     # # Estadísticas post-escalado
#     print("\nEstadísticas después del procesamiento (Train):")
#     print(X_train_final.describe().loc[['mean', 'std', 'min', 'max']])

#     # # Ejemplo de visualización de una columna
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.hist(X_train[:, 0], bins=30, color='blue', alpha=0.7)
#     plt.title("Antes de escalar")
#     plt.subplot(1, 2, 2)
#     plt.hist(X_train_final.iloc[:, 0], bins=30, color='red', alpha=0.7)
#     plt.title("Después de escalar" if usar_escalado else "Sin escalar")
#     plt.show()

#     model = LGBMClassifier(min_data_in_leaf=1, verbosity=-1)
#     model.fit(X_train_final, y_train)

#     # Entrenar y comparar modelos (LazyClassifier)
#     clf = LazyClassifier(verbose=0, ignore_warnings=True)
#     models, predictions = clf.fit(X_train_final, X_test_final, y_train, y_test)
#     models["Dataset"] = os.path.basename(ruta)
#     resultados.append(models)

# # Concatenar todos los resultados y mostrar ordenados por dataset y F1 Score
# df_resultados = pd.concat(resultados)
# df_resultados = df_resultados.sort_values(by=["Dataset", "F1 Score"], ascending=[True, False])
# print(df_resultados)

# # Obtener el mejor modelo (mayor F1 Score) por dataset
# mejores = df_resultados.loc[df_resultados.groupby("Dataset")["F1 Score"].idxmax()]

# # Crear heatmap: filas = datasets, columnas = modelos, valores = F1 Score
# heatmap_df = pd.pivot_table(mejores, index="Dataset", columns="Model", values="F1 Score")

# plt.figure(figsize=(8, 5))
# sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'F1 Score'})
# plt.title("Best model per dataset (F1 Score) - (Scaled scenarios)" if usar_escalado else "Best model per dataset (F1 Score) - (Unscaled scenarios)")
# plt.ylabel("Dataset")
# plt.xlabel("Models")
# plt.tight_layout()
# plt.show()

# # Guardar resultados en CSV
# #df_resultados.to_csv("resultados_lazyclassifier_todos.csv", index=False, encoding='utf-8-sig')