import os
from FlowCytometryTools import FCMeasurement
import pandas as pd

# Ruta de la carpeta que contiene los archivos FCS
carpeta_fcs = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs'

# Lista para almacenar los resultados de los canales de cada archivo
canales = []

# Saber cuántos archivos contienen el canal CCR9
# canal_objetivo = "CCR9 PB450-A"
# contador_canal = 0

# Recorre todos los archivos en la carpeta
for archivo in os.listdir(carpeta_fcs):
    if archivo.endswith('.fcs'):
        # Ruta completa del archivo
        ruta_archivo = os.path.join(carpeta_fcs, archivo)
        
        # Cargar el archivo FCS
        sample = FCMeasurement(ID='Test Sample', datafile=ruta_archivo)
        
        # Extraer los canales del archivo
        canales_archivo = set(sample.channel_names)
        
        # Almacenar el nombre del archivo y los canales detectados
        canales.append({'Archivo': archivo, 'Canales': ', '.join(canales_archivo)})

        # if canal_objetivo in canales_archivo:
        #     contador_canal += 1

# Crear un DataFrame para almacenar los resultados
df_canales = pd.DataFrame(canales)
# print(f"\nEl canal '{canal_objetivo}' está presente en {contador_canal} archivos FCS.")
# Guardar los resultados en un archivo CSV
#df_canales.to_csv('canales_detectados.csv', index=False)

# Para encontrar los canales comunes entre todos los archivos:
# Inicializamos un set con los canales del primer archivo
if canales:
    canales_comunes = set(canales[0]['Canales'].split(', '))
    
    # Recorremos todos los demás archivos para encontrar la intersección de los canales
    for archivo_info in canales[1:]:
        canales_archivo = set(archivo_info['Canales'].split(', '))
        canales_comunes &= canales_archivo  # Intersección
    
    # Imprimir los canales comunes
    print("\nCanales comunes a todos los archivos:")
    print(canales_comunes)
    print(f"Número de canales comunes: {len(canales_comunes)}")

else:
    print("No se encontraron archivos FCS en la carpeta especificada.")
