import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Leer archivo (asegúrate que esté bien codificado)
#df = pd.read_excel(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Entrenamiento\Datos clínicos2.xlsx", engine='openpyxl')
df = pd.read_excel(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Test\Datos demograficos.xlsx", engine='openpyxl')

# Separar el identificador
record_ids = df['Patient_code']
sexo = df['Sexo']

# Definir columnas a procesar
num_cols = ['Edad en el momento de estudio']
cat_cols = ['Grupos de riesgo', 'Sintomas', 'HLA:grupos de riesgo']

# Pipelines por tipo de dato
numeric_pipeline = Pipeline(steps=[
    ('knn_imputer', KNNImputer(n_neighbors=5))
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Transformador combinado
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Aplicar transformación
df_transformed = preprocessor.fit_transform(df)

# Codificar Sexo a 0 y 1 (ajusta los valores según tus datos)
sexo_codificado = sexo.map({'Mujer': 0, 'Hombre': 1})

# Reconstruir DataFrame con columnas originales
all_cols = num_cols + cat_cols
# Reemplazar espacios por guiones bajos en los nombres de columnas
all_cols = [col.replace(" ", "_") for col in all_cols]

df_imputed = pd.DataFrame(df_transformed, columns=all_cols)

# Añadir Record Id y Sexo de vuelta
df_imputed.insert(0, 'Sexo', sexo_codificado)
df_imputed.insert(0, 'Patient_code', record_ids)

# Renombrar todas las columnas para reemplazar espacios por guiones bajos (por si acaso)
df_imputed.columns = [col.replace(" ", "_").replace(":", "_") for col in df_imputed.columns]

# # Decodificar categóricas a texto original
# encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
# df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])

# Ver resultado
print(df_imputed.head())

# Guardar archivo imputado como CSV
#df_imputed.to_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Entrenamiento\Datos clínicos2_imputados.csv", index=False)  # utf-8-sig ayuda a mantener tildes en Excel
df_imputed.to_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Test\Datos demográficos_imputados.csv", index=False)  