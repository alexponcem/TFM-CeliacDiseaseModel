import pandas as pd
import os
import csv
import re

# === PARÁMETROS ===
#input_dir = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables'
#input_dir = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables'
input_dir = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\HighActivationVariables1'
#input_dir = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo\Variables test externo'
#input_dir = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo\HighActivationVariablesTest'
#output_dir = os.path.join(input_dir, 'Variables modificadas')
output_dir = os.path.join(input_dir, 'HAV modificadas')
#output_dir = os.path.join(input_dir, 'Variables modificadas (test externo)')
#output_dir = os.path.join(input_dir, 'HAV modificadas (test externo)')
os.makedirs(output_dir, exist_ok=True)

# === FUNCIONES AUXILIARES ===
# Transforma cada archivo eliminando la primera fila y ajustando el formato
def transformar_y_eliminar_primera(input_file, output_file):
    nombres = []
    valores = []

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                nombres.append(row[0])
                valores.append(row[1])

    if len(nombres) > 1:
        nombres = nombres[1:]
        valores = valores[1:]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(nombres)
        writer.writerow(valores)

# PASO 1: TRANSFORMAR ARCHIVOS INDIVIDUALES
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        try:
            transformar_y_eliminar_primera(input_file, output_file)
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

# PASO 2: AGRUPAR POR PACIENTE
files = [f for f in os.listdir(output_dir) if f.endswith("_features.csv")]
patients_data = {}

for file in files:
    match = re.match(r"(.+?)_(PRE|POST1)", file)
    if not match:
        continue
    patient_code = match.group(1)
    time_point = match.group(2)
    df = pd.read_csv(os.path.join(output_dir, file))

    if patient_code not in patients_data:
        patients_data[patient_code] = {}
    patients_data[patient_code][time_point] = df

# PASO 3: GENERAR DATASETS DIFERENCIA Y CONCATENACIÓN
diff_data = []
concat_data = []

for patient_code, data in patients_data.items():
    if "PRE" not in data or "POST1" not in data:
        print(f"[Omitido] {patient_code} no tiene ambos archivos PRE y POST1.")
        continue

    df_pre = data["PRE"].copy()
    df_post = data["POST1"].copy()

    if df_pre.shape[0] != 1 or df_post.shape[0] != 1:
        print(f"[Omitido] {patient_code} tiene múltiples filas.")
        continue

    # ===== DIFERENCIA =====
    common_cols = df_pre.columns.intersection(df_post.columns)
    df_pre_common = df_pre[common_cols]
    df_post_common = df_post[common_cols]
    df_diff = df_post_common - df_pre_common
    df_diff.insert(0, "Patient_code", patient_code)

    if "Etiqueta" in df_pre.columns:
        df_diff["Etiqueta"] = df_pre["Etiqueta"].values[0]

    diff_data.append(df_diff)

    # ===== CONCATENACIÓN =====
    target_val = None
    if "Etiqueta" in df_pre.columns:
        target_val = df_pre["Etiqueta"].values[0]
        df_pre = df_pre.drop(columns=['Etiqueta'])
    if "Etiqueta" in df_post.columns:
        df_post = df_post.drop(columns=['Etiqueta'])

    df_pre.columns = [f"{col}_pre" for col in df_pre.columns]
    df_post.columns = [f"{col}_post" for col in df_post.columns]
    df_concat = pd.concat([df_pre, df_post], axis=1)
    df_concat.insert(0, "Patient_code", patient_code)
    if target_val is not None:
        df_concat["Etiqueta"] = target_val

    concat_data.append(df_concat)


# PASO 4: GUARDAR ARCHIVOS FINALES
df_diff_final = pd.concat(diff_data, ignore_index=True)
df_concat_final = pd.concat(concat_data, ignore_index=True)

# PASO 5: FUSIÓN CON DATOS CLÍNICOS 
# Cargar los datos clínicos imputados (asegúrarse que el archivo existe y tiene la columna 'Patient_code')
df_clinical = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Entrenamiento\Datos clínicos2_imputados.csv")  # <-- cambia esto
#df_clinical = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Test\Datos demográficos_imputados.csv")  # <-- cambia esto

# Fusionar con los datasets generados
df_diff_merged = pd.merge(df_diff_final, df_clinical, on="Patient_code", how="inner")
df_concat_merged = pd.merge(df_concat_final, df_clinical, on="Patient_code", how="inner")

# # Guardar las versiones fusionadas
# df_diff_merged.to_csv(os.path.join(output_dir, "dataset_diferencias_pacientes_CON_CLINICOS.csv"), index=False, encoding='utf-8-sig')
# df_concat_merged.to_csv(os.path.join(output_dir, "dataset_concatenado_pacientes_CON_CLINICOS.csv"), index=False, encoding='utf-8-sig')

# PASO FINAL: FUSIÓN CON DIAGNÓSTICO DESDE CSV
# Cargar el archivo CSV con el diagnóstico de cada paciente (debe tener columnas 'Patient_code' y 'Etiqueta')
df_target = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Entrenamiento\Target.csv")  # <-- ajusta la ruta y nombre
#df_target = pd.read_csv(r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Datos clínicos y demográficos\Test\Target_test.csv")  # <-- ajusta la ruta y nombre

# Fusionar con los datasets ya fusionados con datos clínicos
df_diff_merged = pd.merge(df_diff_merged, df_target[['Patient_code', 'Etiqueta']], on="Patient_code", how="left")
df_concat_merged = pd.merge(df_concat_merged, df_target[['Patient_code', 'Etiqueta']], on="Patient_code", how="left")

# Guardar las versiones finales con diagnóstico
df_diff_merged.to_csv(os.path.join(output_dir, "dataset_diferencias_DC.csv"), index=False) #Cambiar nombre si es necesario
df_concat_merged.to_csv(os.path.join(output_dir, "dataset_concatenado_DC.csv"), index=False) #Cambiar nombre si es necesario

print("✅ Archivos fusionados generados:")
print(" - dataset_diferencias_DC.csv")
print(" - dataset_concatenado_DC.csv")

# print("✅ Archivos generados:")
# print(" - dataset_diferencias_HA_pacientes.csv")
# print(" - dataset_concatenado_HA_pacientes.csv")