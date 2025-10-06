from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import pandas as pd

# DATASET CON DATOS CLÍNICOS
# Dataset de diferencias entre pacientes con datos clínicos
#ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_diferencias_DC.csv"
# Dataset concatenado con datos clínicos
ruta = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_concatenado_DC.csv"
final_df = pd.read_csv(ruta)

# Convertir la columna "Etiqueta" a numérica e imprimir conteo
final_df["Etiqueta"] = final_df["Etiqueta"].map({"Negativo": 0, "Positivo": 1})
print(final_df["Etiqueta"].value_counts())

# Cargar datos (asegurarse de que 'final_df' ya esté definido)
X = final_df.drop(["Etiqueta", "Patient_code"], axis=1)
y = final_df["Etiqueta"]

# Umbral de varianza (ajustarlo según necesidad: 0.01 para eliminar casi constantes)
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

model = LGBMClassifier(min_data_in_leaf=1, verbosity=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("LightGBM personalizado:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Entrenar y comparar modelos
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Mostrar resultados
print(models)