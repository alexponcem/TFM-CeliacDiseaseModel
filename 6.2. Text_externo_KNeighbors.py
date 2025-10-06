import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# =========================================
# 1. PREPARAR DATOS
# =========================================
# Aquí se deben cargar los datasets ya preprocesados
df_all = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv")
df_new = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo\Variables test externo\Variables filtradas\dataset_concatenado_DC_fil_test.csv")

# Variables y etiquetas
X_train = df_all.drop(columns=["Etiqueta", "Patient_code"])
y_train = df_all["Etiqueta"].map({"Negativo": 0, "Positivo": 1})

X_test = df_new.drop(columns=["Etiqueta", "Patient_code"])
y_test = df_new["Etiqueta"].map({"Negativo": 0, "Positivo": 1})

# Imputar solo en X (no en y)
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# =========================================
# 2. DEFINIR MODELOS CANDIDATOS
# =========================================
models = {
     "KNeighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": 3,#[3, 5, 7, 9, 11],
            "weights": "distance",#["uniform", "distance"],
            "algorithm": "brute", #["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size":40, #[20, 30, 40],
            "p": 1#[1, 2]
        }
    },
}
# Resultados de CV (rellenar con las métricas ya calculadas)
cv_results = {
    "KNeighbors": {"bal_acc": 0.684, "f1": 0.703, "auc": 0.670}
}

# =========================================
# 3. FUNCIÓN DE VALIDACIÓN EXTERNA
# =========================================
def external_validation(models, X_train, y_train, X_test, y_test, cv_results=None):
    results = []

    for name, model_dict in models.items():
        # Obtener el modelo
        model = model_dict["model"]

        # Configurar hiperparámetros
        model.set_params(**model_dict["params"])

        # entrenar modelo en todo el dataset original
        model.fit(X_train, y_train)
        
        # predicciones sobre las 19 nuevas muestras
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
    # # Mostrar cómo predice el modelo para cada muestra
    #     df_pred = pd.DataFrame({
    #         "Patient_code": df_new["Patient_code"].values if "Patient_code" in df_new.columns else np.arange(len(y_test)),
    #         "Etiqueta real": y_test.values,
    #         #"Probabilidad predicha": y_proba,
    #         "Predicción final": y_pred
    #     })
    #     print("\nPredicciones individuales del modelo:")
    #     print(df_pred)

        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]

        best_t = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        print(f"Best threshold for F1: {best_t}, F1 ={best_f1:.3f}")

        # métricas
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # guardar resultados
        row = {
            "Model": name,
            "Balanced accuracy (Test)": round(bal_acc, 3),
            "F1-score (Test)": round(f1, 3),
            "ROC AUC (Test)": round(auc, 3)
        }
        
        if cv_results and name in cv_results:
            row["Balanced accuracy (CV)"] = cv_results[name].get("bal_acc")
            row["F1-score (CV)"] = cv_results[name].get("f1")
            row["ROC AUC (CV)"] = cv_results[name].get("auc")
        
        results.append(row)

    return pd.DataFrame(results)

# =========================================
# 4. EJECUTAR Y MOSTRAR RESULTADOS
# =========================================
df_results = external_validation(models, X_train, y_train, X_test, y_test, cv_results=cv_results)
cols = ["Model", 
        "Balanced accuracy (CV)", "Balanced accuracy (Test)", 
        "F1-score (CV)", "F1-score (Test)", 
        "ROC AUC (CV)", "ROC AUC (Test)"]
df_results = df_results.reindex(columns=cols)
print(df_results)

