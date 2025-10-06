import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

# ----------------- Configuración mejorada -----------------
datasets = [
    {"name": "Filtered difference", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_diferencias_DC_fil.csv", "scale": True},
    {"name": "Filtered concatenation", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv", "scale": True},
    {"name": "Filtered difference (no scaling)", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_diferencias_DC_fil.csv", "scale": False},
    {"name": "Filtered concatenation (no scaling)", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv", "scale": False},
]

# Hiperparámetros más completos para cada modelo
models = {
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0, 1, 5],
            "min_child_weight": [1, 5, 10],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2]
        }
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False]
        }
    },
    "ExtraTrees": {
        "model": ExtraTreesClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "max_features": ['sqrt', 'log2']
        }
    },
    "NuSVC": {
        "model": NuSVC(probability=True, random_state=42),
        "params": {
            "nu": [0.3, 0.5, 0.7],
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNeighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2]
        }
    },
    "RidgeClassifierCV": {
        "model": RidgeClassifierCV(),
        "params": {
            "alphas": [[0.1, 1.0, 10.0, 100.0]],
            "fit_intercept": [True, False],
            "class_weight": [None, "balanced"]
        }
    },
    "PassiveAggressive": {
        "model": PassiveAggressiveClassifier(random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 5000],
            "tol": [1e-4, 1e-3, 1e-2],
            "loss": ["hinge", "squared_hinge"],
            "class_weight": [None, "balanced"]
        }
    }
}

# Configuración de semillas aleatorias para probar
random_seeds = [42, 123, 456, 789, 101112]  # 5 semillas diferentes
results = []
fold_results = [] # Lista para almacenar resultados de cada fold

# ----------------- Bucle de evaluación mejorado -----------------

for d in datasets:
    print(f"\n📊 Evaluando dataset: {d['name']}")
    df = pd.read_csv(d["path"])
    df["Etiqueta"] = df["Etiqueta"].map({"Negativo": 0, "Positivo": 1})
    X = df.drop(columns=["Etiqueta", "Patient_code"])
    y = df["Etiqueta"]

    for seed in random_seeds:
        print(f"\n🔁 Usando random seed: {seed}")

        # Imputación de valores faltantes con mediana
        imputer = SimpleImputer(strategy="median")
        X_proc = imputer.fit_transform(X)

        # Filtrado por varianza
        selector = VarianceThreshold(threshold=0.01)
        X_proc = selector.fit_transform(X_proc)

        # Escalado (si aplica)
        if d["scale"]:
            scaler = RobustScaler()
            X_proc = scaler.fit_transform(X_proc)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for model_name, config in models.items():
            print(f"  🔍 Model: {model_name}")
            start = time.time()

            grid = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=50,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=0,
                random_state=seed
            )
            grid.fit(X_proc, y)
            best_model = grid.best_estimator_

            # Validación cruzada con el mejor modelo
            f1_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='f1')
            acc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='accuracy')
            bal_acc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='balanced_accuracy')
            roc_auc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='roc_auc')
            end = time.time()

            for i, f1 in enumerate(f1_scores):
                fold_results.append({
                    "Dataset": d["name"],
                    "Random_Seed": seed,
                    "Model": model_name,
                    "Fold": f"fold_{i+1}",
                    "F1-Score": f1
                })

            results.append({
                "Dataset": d["name"],
                "Random_Seed": seed,
                "Model": model_name,
                "BestParams": str(grid.best_params_),
                "F1_CV_Mean": np.mean(f1_scores),
                "F1_CV_Std": np.std(f1_scores),
                "Accuracy_CV_Mean": np.mean(acc_scores),
                "Accuracy_CV_Std": np.std(acc_scores),
                "BalancedAcc_CV_Mean": np.mean(bal_acc_scores),
                "BalancedAcc_CV_Std": np.std(bal_acc_scores),
                "ROC_AUC_CV_Mean": np.mean(roc_auc_scores),
                "ROC_AUC_CV_Std": np.std(roc_auc_scores),
                "Time_sec": round(end - start, 2)
            })


# Crear DataFrame de resultados por fold
fold_results_df = pd.DataFrame(fold_results)

# Añadir columna 'Scaled' al DataFrame de folds
fold_results_df["Scaled"] = fold_results_df["Dataset"].str.contains("no_scaling").map({True: "No Escalado", False: "Escalado"})

# Gráfico SOLO para datasets escalados
g1 = sns.catplot(
    data=fold_results_df[fold_results_df["Scaled"] == "Escalado"],
    x="Fold",
    y="F1-Score",
    hue="Model",
    col="Dataset",
    kind="bar",
    ci=None,
    height=5,
    aspect=1.2
)
g1.fig.subplots_adjust(top=0.85)
#g1.fig.suptitle("F1-score por fold y modelo (Datasets Escalados)")
plt.show()

# Gráfico SOLO para datasets NO escalados
g2 = sns.catplot(
    data=fold_results_df[fold_results_df["Scaled"] == "No Escalado"],
    x="Fold",
    y="F1-Score",
    hue="Model",
    col="Dataset",
    kind="bar",
    ci=None,
    height=5,
    aspect=1.2
)
g2.fig.subplots_adjust(top=0.85)
#g2.fig.suptitle("F1-score por fold y modelo (Datasets No Escalados)")
plt.show()

# ----------------- Análisis de resultados -----------------
results_df = pd.DataFrame(results)

# Guardar resultados completos
#results_df.to_csv("resultados_modelos_cv.csv", index=False)
results_df.to_csv("resultados_modelos_cvNUEVO.csv", index=False)
print("\n✅ Resultados guardados como 'resultados_modelos_cv.csv'")

# Análisis por semilla
print("\n📊 Resumen de resultados por semilla:")
print(results_df.groupby("Random_Seed")[["F1_CV_Mean"]].mean())

# Análisis general
print("\n📊 Resumen general de resultados:")
print(results_df.groupby(["Dataset", "Model"])[["F1_CV_Mean"]].mean())

# Visualización
plt.figure(figsize=(14, 8))
box = sns.boxplot(
    data=results_df, 
    x="Dataset", 
    y="F1 CV Mean", 
    hue="Model",
    palette="Set2",  # Mejor paleta de colores
    linewidth=1.5,   # Líneas más definidas
    width=0.7        # Control del ancho de las cajas
)

# Mejoras específicas para el eje X:
plt.title("F1 score (CV) Distribution by Model and Dataset (multiple seeds)", pad=20, fontsize=14)
plt.xlabel("Dataset configuration", fontsize=12, labelpad=15)
plt.ylabel("F1_CV_Mean", fontsize=12, labelpad=10)

# Rotación y alineación de etiquetas
box.set_xticklabels(
    box.get_xticklabels(),
    rotation=15,
    ha='right',
    rotation_mode='anchor',
    fontsize=12
)

# Ajustar posición de la leyenda
plt.legend(
    title='Modelos',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

# Línea horizontal de referencia
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Ajustes finales
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Convierte la columna BestParams de string a dict
results_df["BestParams_dict"] = results_df["BestParams"].apply(ast.literal_eval)

# Extrae los hiperparámetros a columnas separadas
params_df = results_df[["Dataset", "Model", "BestParams_dict"]].copy()
params_expanded = params_df["BestParams_dict"].apply(pd.Series)
params_full = pd.concat([params_df[["Dataset", "Model"]], params_expanded], axis=1)

# Tabla de los hiperparámetros más frecuentes por modelo y dataset
param_mode = (
    params_full
    .groupby(["Dataset", "Model"])
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
)

print("\n📋 Hiperparámetros óptimos más frecuentes por modelo y dataset:")
print(param_mode)

# # Si solo quieres el más frecuente por modelo (sin distinguir dataset):
# param_mode_model = (
#     params_full
#     .groupby("Model")
#     .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
#     .reset_index()
# )
# print("\n📋 Hiperparámetros óptimos más frecuentes por modelo (todos los datasets):")
# print(param_mode_model)

# Guardar a CSV si lo deseas
#param_mode.to_csv("hiperparametros_frecuentes_por_modelo_y_dataset.csv", index=False)
#param_mode.to_csv("hiperparametros_frecuentes_por_modelo_y_datasetNUEVO.csv", index=False)
# param_mode_model.to_csv("hiperparametros_frecuentes_por_modelo.csv", index=False)