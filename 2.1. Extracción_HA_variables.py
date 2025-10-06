import os
import time
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, gaussian_kde, iqr
from FlowCytometryTools import FCMeasurement

# Canales definidos como necesarios para la coexpresión activada
ACTIVATION_CHANNELS = ['CD103 FITC-A','B7 PE-A','FSC-A','SSC-A','CD38 PC7-A','CD8 APC-A']
#ACTIVATION_CHANNELS = ['FSC-Width', 'CD103 FITC-H', 'CD38 PC7-H', 'CD8 APC-H', 'FSC-H', 'Time', 'CD38 PC7-A', 'B7 PE-H', 'CD103 FITC-A', 'SSC-A', 'B7 PE-A','SSC-H', 'FSC-A', 'CD8 APC-A']
COMMON_CHANNELS = set(ACTIVATION_CHANNELS)  # o se pueden agregar más para extraer más features

# Función para extraer características estadísticas de un DataFrame
def extract_distribution_features(df: pd.DataFrame) -> pd.Series:
    features = {}
    for col in df.columns.intersection(COMMON_CHANNELS):
        data = df[col].dropna()
        if len(data) == 0:
            continue

        # Tendencia central
        features[f'{col}_mean'] = np.mean(data)
        features[f'{col}_median'] = np.median(data)
        features[f'{col}_mode'] = data.mode().iloc[0] if not data.mode().empty else np.nan
        
        # Dispersión
        features[f'{col}_std'] = np.std(data, ddof=1)
        features[f'{col}_var'] = np.var(data, ddof=1)
        features[f'{col}_range'] = np.ptp(data)
        features[f'{col}_iqr'] = iqr(data)
        features[f'{col}_cv'] = features[f'{col}_std'] / features[f'{col}_mean'] if features[f'{col}_mean'] != 0 else np.nan
        
        # Forma de la distribución
        features[f'{col}_skew'] = skew(data)
        features[f'{col}_kurtosis'] = kurtosis(data)

        # Cuantiles
        features[f'{col}_min'] = np.min(data)
        features[f'{col}_q05'] = np.percentile(data, 5)
        features[f'{col}_q25'] = np.percentile(data, 25)
        features[f'{col}_q75'] = np.percentile(data, 75)
        features[f'{col}_q95'] = np.percentile(data, 95)
        features[f'{col}_max'] = np.max(data)

        # Entropía
        hist, _ = np.histogram(data, bins='auto', density=True)
        if hist.sum() > 0 and not np.isnan(hist).any():
            features[f'{col}_entropy'] = entropy(hist + 1e-8)
        else:
            features[f'{col}_entropy'] = np.nan

        # Outliers
        lower = features[f'{col}_q25'] - 1.5 * features[f'{col}_iqr']
        upper = features[f'{col}_q75'] + 1.5 * features[f'{col}_iqr']
        features[f'{col}_num_outliers'] = np.sum((data < lower) | (data > upper))

        # Modos
        try:
            kde = gaussian_kde(data)
            xs = np.linspace(np.min(data), np.max(data), 200)
            density = kde(xs)
            peaks = np.where((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]))[0]
            features[f'{col}_num_modes'] = len(peaks)
        except:
            features[f'{col}_num_modes'] = np.nan
    return pd.Series(features)

# Función para filtrar la zona de alta activación con umbrales personalizados
def get_multi_marker_activation_zone(df: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, dict]:
    
    #Filtra la zona de alta activación aplicando un porcentaje distinto por canal, según las indicaciones clínicas.

    # Umbrales personalizados por canal
    custom_thresholds = {
        'B7 PE-A': 0.05,
        'CD103 FITC-A': 0.10,
        'CD8 APC-A': 0.25,
        'CD38 PC7-A': 0.10,
        # FSC-A y SSC-A no se filtran, se incluyen todos los valores
    }

    thresholds = {}
    condition = pd.Series([True] * len(df))

    # Aplicar umbrales personalizados
    for channel in channels:
        if channel not in df.columns:
            continue

        if channel in custom_thresholds:
            pct = custom_thresholds[channel]
            threshold = np.percentile(df[channel], 100 * (1 - pct))
            thresholds[channel] = threshold
            condition &= df[channel] >= threshold
        else:
            # No se aplica umbral: se incluye todo
            thresholds[channel] = None

    filtered_df = df[condition]
    return filtered_df, thresholds

# Función principal para procesar un par de archivos FCS (PRE y POST1)
def process_pair_fcs_files(pre_path: str, post_path: str, activation_channels=None):
    if activation_channels is None:
        activation_channels = ACTIVATION_CHANNELS

    pre_df = FCMeasurement(ID='PRE', datafile=pre_path).data
    post_df = FCMeasurement(ID='POST1', datafile=post_path).data

    pre_filtered, thresholds = get_multi_marker_activation_zone(pre_df, activation_channels)

    post_condition = pd.Series([True] * len(post_df))
    for ch, thr in thresholds.items():
        if thr is not None and ch in post_df.columns:
            post_condition &= post_df[ch] >= thr
    post_filtered = post_df[post_condition]

    pre_features = extract_distribution_features(pre_filtered)
    post_features = extract_distribution_features(post_filtered)

    return pre_features, post_features

# Función para encontrar y emparejar archivos PRE y POST1 en una carpeta
def get_patient_pairs(folder_path):
    
    #Busca archivos que contengan _PRE o _POST1, y los empareja por ID base.
    
    fcs_files = [f for f in os.listdir(folder_path) if f.endswith('.fcs')]
    patients = {}

    for file in fcs_files:
        if '_PRE' in file or '_POST1' in file:
            # Extrae el ID base respetando mayúsculas/minúsculas
            if '_PRE' in file:
                patient_id = file.split('_PRE')[0]
                patients.setdefault(patient_id, {})['pre'] = file
            elif '_POST1' in file:
                patient_id = file.split('_POST1')[0]
                patients.setdefault(patient_id, {})['post'] = file

    return [
        (pid, os.path.join(folder_path, files['pre']), os.path.join(folder_path, files['post']))
        for pid, files in patients.items()
        if 'pre' in files and 'post' in files
    ]

# Función principal para ejecutar el procesamiento de todos los pares en una carpeta
def main():
    folder_path = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs'
    output_folder = os.path.join(folder_path, 'HighActivationVariables1')
    
    #Test externo
    #folder_path = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo'
    #output_folder = os.path.join(folder_path, 'HighActivationVariablesTest')

    os.makedirs(output_folder, exist_ok=True)

    patient_pairs = get_patient_pairs(folder_path)

    if not patient_pairs:
        print("❌ No se encontraron pares PRE/POST1 válidos.")
        return

    for patient_id, pre_path, post_path in patient_pairs:
        try:
            print(f"\n🔄 Procesando: {patient_id}")
            start = time.time()
            
            pre_features, post_features = process_pair_fcs_files(pre_path, post_path)
            
            pre_output = os.path.join(output_folder, f"{patient_id}_PRE_activated_features.csv")
            post_output = os.path.join(output_folder, f"{patient_id}_POST1_activated_features.csv")

            pre_features.to_csv(pre_output)
            post_features.to_csv(post_output)

            print(f"✔ Completado en {time.time() - start:.2f}s")
        except Exception as e:
            print(f"❌ Error con {patient_id}: {e}")


if __name__ == '__main__':
    main()
