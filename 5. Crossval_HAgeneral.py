import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
import ast
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

# ----------------- Configuración mejorada -----------------
#datasets = [
# #    {"name": "Dif_filtrado", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables\HAV modificadas\dataset_diferencias_HA_DC.csv", "scale": True},
# #    {"name": "Concat_filtrado", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables\HAV modificadas\dataset_concatenado_HA_DC.csv", "scale": True},
#     {"name": "Dif_filtrado_no_scaling", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables\HAV modificadas\dataset_diferencias_HA_DC.csv", "scale": False},
#     {"name": "Concat_filtrado_no_scaling", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables\HAV modificadas\dataset_concatenado_HA_DC.csv", "scale": False},
# ]

# Datasets with clinical data and different thresholds
datasets = [
#    {"name": "Dif_filtrado", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables1\HAV modificadas\dataset_diferencias_DC.csv", "scale": True},
#    {"name": "Concat_filtrado", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables1\HAV modificadas\dataset_concatenado_DC.csv", "scale": True},
    {"name": "Dif_filtrado_no_scaling", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables1\HAV modificadas\dataset_diferencias_DC.csv", "scale": False},
#    {"name": "Concat_filtrado_no_scaling", "path": r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables1\HAV modificadas\dataset_concatenado_DC.csv", "scale": False},
]

# Hiperparámetros más completos para cada modelo
models = {
    # "BernoulliNB": {
    #     "model": BernoulliNB(),
    #     "params": {
    #         "alpha": [0.5, 1.0, 2.0],
    #         "fit_prior": [True, False]
    #     }
    # },
    # "DecisionTree": {
    #     "model": DecisionTreeClassifier(random_state=42),
    #     "params": {
    #         "max_depth": [None, 5, 10, 20],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4]
    #     }
    # },
    # "GaussianNB": {
    #     "model": GaussianNB(),
    #     "params": {}  # No tiene hiperparámetros principales para optimizar
    # },
    # "RandomForest": {
    #     "model": RandomForestClassifier(random_state=42),
    #     "params": {
    #         "n_estimators": [100, 200, 300],
    #         "max_depth": [None, 5, 10, 20],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4],
    #         "bootstrap": [True, False],
    #         "max_features": ['sqrt', 'log2']
    #     }
    # },
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
    #  "KNeighbors": {
    #     "model": KNeighborsClassifier(),
    #     "params": {
    #         "n_neighbors": [3, 5, 7, 9, 11],
    #         "weights": ["uniform", "distance"],
    #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    #         "leaf_size": [20, 30, 40],
    #         "p": [1, 2]
    #     }
    # },
    # "SGDClassifier": {
    #     "model": SGDClassifier(random_state=42),
    #     "params": {
    #         "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
    #         "penalty": ["l2", "l1", "elasticnet"],
    #         "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    #         "max_iter": [1000, 2000, 5000],
    #         "tol": [1e-4, 1e-3, 1e-2],
    #         "fit_intercept": [True, False],
    #         "learning_rate": ["optimal", "invscaling", "adaptive"],
    #         "eta0": [0.001, 0.01, 0.1],
    #         "power_t": [0.5, 0.7, 0.9],
    #         "class_weight": [None, "balanced"]
    #     }
    # },
    # "AdaBoost": {
    #     "model": AdaBoostClassifier(random_state=42),
    #     "params": {
    #         "n_estimators": [50, 100, 200],
    #         "learning_rate": [0.01, 0.1, 1.0],
    #     }
    # },
    # "ExtraTrees": {
    #     "model": ExtraTreesClassifier(random_state=42),
    #     "params": {
    #         "n_estimators": [100, 200, 300],
    #         "max_depth": [None, 5, 10, 20],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4],
    #         "bootstrap": [True, False],
    #         "max_features": ['sqrt', 'log2']
    #     }
    # },
    # "XGBoost": {
    #     "model": XGBClassifier(eval_metric='logloss', random_state=42),
    #     "params": {
    #         "n_estimators": [100, 200],
    #         "max_depth": [3, 6, 10],
    #         "learning_rate": [0.01, 0.1, 0.2],
    #         "subsample": [0.7, 0.8, 1.0],
    #         "colsample_bytree": [0.7, 0.8, 1.0],
    #         "gamma": [0, 1, 5],
    #         "min_child_weight": [1, 5, 10],
    #         "reg_alpha": [0, 0.1, 1],
    #         "reg_lambda": [1, 1.5, 2]
    #     }
    #  },
}

# Configuración de semillas aleatorias para probar
random_seeds = [42]#, 123, 456, 789, 101112]  # 5 semillas diferentes

results = []

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
        
        #ESTO ES NUEVO
        selected_columns = X.columns[selector.get_support()].tolist()

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for model_name, config in models.items():
            print(f"  🔍 Model: {model_name}")
            start = time.time()

            grid = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=50,  # Reduce el número de iteraciones para mayor rapidez
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=seed
            )
            grid.fit(X_proc, y)
            best_model = grid.best_estimator_

            # Calcula SHAP solo para modelos compatibles (árboles y lineales)
        # X_proc to dataframe with selected columns
        X_proc = pd.DataFrame(X_proc, columns=selected_columns)
        try:
            if X_proc.shape[1] > 1:
                explainer = None
                if model_name in ["RandomForest", "ExtraTrees", "XGBoost", "Bagging", "DecisionTree"]:#, "AdaBoost"]:
                    # Para Bagging, usa el primer estimador base
                    if model_name == "Bagging":
                        base_estimator = best_model.estimators_[0]
                        print("Tipo de estimador base:", type(base_estimator))
                        print("Columnas de X_proc:", X_proc.columns.tolist())
                        print("Shape de X_proc:", X_proc.shape)
                        explainer = shap.TreeExplainer(base_estimator)
                        shap_values = explainer.shap_values(X_proc)
                        shap.summary_plot(shap_values, X_proc, max_display=20)
                        plt.title(f"SHAP Summary - {model_name} ({d['name']})")
                        plt.tight_layout()
                        plt.savefig(f"shap_summary_{model_name}_{d['name']}.png")
                        plt.close()
                    # elif model_name == "AdaBoost":
                    #     base_estimator = best_model.estimator
                    #     explainer = shap.TreeExplainer(base_estimator)
                    elif model_name == "XGBoost":
                        explainer = shap.TreeExplainer(best_model)
                    else:
                        explainer = shap.TreeExplainer(best_model)
                # elif model_name in ["LogisticRegression"]:
                #     explainer = shap.LinearExplainer(best_model, X_proc)
                # elif model_name == "KNeighbors":
                #     try:
                #         background = shap.sample(X_proc, 100) if X_proc.shape[0] > 100 else X_proc
                #         explainer = shap.KernelExplainer(best_model.predict_proba, background)
                #         shap_values = explainer.shap_values(X_proc)
                #         shap.summary_plot(shap_values[1], X_proc, max_display=20)
                #         plt.title(f"SHAP Summary - {model_name} ({d['name']})")
                #         plt.tight_layout()
                #         plt.savefig(f"shap_summary_{model_name}_{d['name']}.png")
                #         plt.close()
                #     except Exception as e:
                #         print(f"SHAP no soportado para {model_name}: {e}")
                if explainer is not None:
                    shap_values = explainer.shap_values(X_proc)
                    # Para XGBoost y modelos de árbol, shap_values es 2D
                    shap.summary_plot(shap_values, X_proc, max_display=20)
                    plt.title(f"SHAP Summary - {model_name} ({d['name']})")
                    plt.tight_layout()
                    plt.savefig(f"shap_summary_{model_name}_{d['name']}.png")
                    plt.close()
                # if explainer is not None:
                #     shap_values = explainer.shap_values(X_proc)
                #     # Solo si hay más de una variable, plotea
                #     if isinstance(shap_values, list):
                #         shap.summary_plot(shap_values[:,:,1], X_proc, max_display=min(40, X_proc.shape[1]))
                #     else:
                #         shap.summary_plot(shap_values[:, :, 1], X_proc, max_display=20, show=False)
                #     plt.title(f"SHAP Summary - {model_name} ({d['name']})")
                #     plt.tight_layout()
                #     plt.savefig(f"shap_summary_{model_name}_{d['name']}.png")
                #     plt.close()
            else:
                print(f"SHAP: Solo hay {X_proc.shape[1]} variable(s), no se genera gráfico para {model_name}.")
        except Exception as e:
            print(f"SHAP no soportado para {model_name}: {e}")

            # Validación cruzada con el mejor modelo
            f1_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='f1')
            acc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='accuracy')
            bal_acc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='balanced_accuracy')
            roc_auc_scores = cross_val_score(best_model, X_proc, y, cv=cv, scoring='roc_auc')
            end = time.time()

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

# ----------------- Análisis de resultados -----------------
results_df = pd.DataFrame(results)

# Guardar resultados completos
#results_df.to_csv("resultados_modelosHA_cv.csv", index=False)
print("\n✅ Resultados guardados como 'resultados_modelosHA_cv.csv'")

# # Solución rápida: inspecciona las columnas y las primeras filas
# print("Columnas en results_df:", results_df.columns.tolist())
# print("Primeras filas:\n", results_df.head())

# # Análisis por semilla
# print("\n📊 Resumen de resultados por semilla:")
# print(results_df.groupby("Random_Seed")[["F1_CV_Mean"]].mean())

# # Análisis general
# print("\n📊 Resumen general de resultados:")
# print(results_df.groupby(["Dataset", "Model"])[["F1_CV_Mean"]].mean())

# # Visualización
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=results_df, x="Model", y="F1_CV_Mean", hue="Dataset")
# plt.title("Distribución de F1 Score (CV) por Modelo y Dataset (varias semillas)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Convierte la columna BestParams de string a dict
# results_df["BestParams_dict"] = results_df["BestParams"].apply(ast.literal_eval)

# # Extrae los hiperparámetros a columnas separadas
# params_df = results_df[["Dataset", "Model", "BestParams_dict"]].copy()
# params_expanded = params_df["BestParams_dict"].apply(pd.Series)
# params_full = pd.concat([params_df[["Dataset", "Model"]], params_expanded], axis=1)

# # Encuentra la combinación de hiperparámetros más frecuente (moda) por modelo y dataset
# param_mode = (
#     params_full
#     .groupby(["Dataset", "Model"])
#     .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
#     .reset_index()
# )

# # Guarda la tabla a CSV
# #param_mode.to_csv("hiperparametros_optimos_mas_frecuentesHA.csv", index=False)
# print("\n✅ Hiperparámetros óptimos más frecuentes guardados como 'hiperparametros_optimos_mas_frecuentes.csv'")