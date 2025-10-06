import pandas as pd
import os

# Define los prefijos válidos
column_prefixes = [
    "Patient_code",
    "CD103 FITC-A",
    "B7 PE-A",
    "FSC-A",
    "SSC-A",
    "CD38 PC7-A",
    "CD8 APC-A",
    "Etiqueta",
    "Sexo",
    "Edad_en_el_momento_de_estudio",
    "Grupos_de_riesgo",
    "Sintomas",
    "HLA_grupos_de_riesgo"
]

# Función para filtrar columnas por prefijo
def filtrar_columnas_por_prefijo(df, prefijos):
    columnas_filtradas = [col for col in df.columns if any(col.startswith(pref) for pref in prefijos)]
    return df[columnas_filtradas]

# Cambia esta ruta al archivo que desees procesar
#Archivo de diferencias entre pacientes
#ruta_csv = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\Variables modificadas\dataset_diferencias_pacientes.csv"
#ruta_guardar=r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\Variables modificadas\Variables filtradas\dataset_dif_filtrado.csv"
#Archivo concatenado
# ruta_csv = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\Variables modificadas\dataset_concatenado_pacientes.csv"
# ruta_guardar=r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\Variables modificadas\Variables filtradas\dataset_concat_filtrado.csv"

# DATOS CLÍNICOS
# Archivo de diferencias Con datos clínicos
# ruta_csv = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_diferencias_DC.csv"
# ruta_guardar=r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_diferencias_DC_fil.csv"
# Archivo concatenado Con datos clínicos
#ruta_csv = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables\Variables modificadas\dataset_concatenado_DC.csv"
#ruta_guardar=r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Variables filtradas\dataset_concatenado_DC_fil.csv"

# # TEST EXTERNO
ruta_csv = r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo\Variables test externo\Variables modificadas (test externo)\dataset_concatenado_Test.csv"
ruta_guardar=r"C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo\Variables test externo\Variables filtradas\dataset_concatenado_DC_fil_test.csv"

df = pd.read_csv(ruta_csv)

# Filtrar columnas
df_filtrado = filtrar_columnas_por_prefijo(df, column_prefixes)

# Guardar nuevo archivo CSV si lo deseas
df_filtrado.to_csv(ruta_guardar, index=False)
print("Columnas filtradas guardadas en archivo_filtrado.csv")
