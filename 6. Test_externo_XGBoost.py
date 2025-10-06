import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC
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
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "min_child_weight": 1,
            "reg_alpha": 0,
            "reg_lambda": 1.5,
        }
    },
    "NuSVC": {
        "model": NuSVC(probability=True, random_state=42),
        "params": {
            "nu": 0.5,
            "kernel": "rbf",
            "gamma": "scale"
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "entropy"
        }
    }
}
# Resultados de CV (rellenar con las métricas ya calculadas)
cv_results = {
    "AdaBoost": {"bal_acc": 0.774, "f1": 0.773, "auc": 0.828},
    "XGBoost": {"bal_acc": 0.790, "f1": 0.788, "auc": 0.781},
    "NuSVC": {"bal_acc": 0.557, "f1": 0.535, "auc": 0.539},
    "DecisionTree": {"bal_acc": 0.744, "f1": 0.721, "auc": 0.752}
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