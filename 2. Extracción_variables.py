import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, gaussian_kde, iqr
from FlowCytometryTools import FCMeasurement
import os
import time

# Canales comunes definidos manualmente
COMMON_CHANNELS = {
    'FSC-Width', 'CD103 FITC-H', 'CD38 PC7-H', 'CD8 APC-H', 'FSC-H', 'Time',
    'CD38 PC7-A', 'B7 PE-H', 'CD103 FITC-A', 'SSC-A', 'B7 PE-A',
    'SSC-H', 'FSC-A', 'CD8 APC-A'
}

# Función para extraer características de distribución de un DataFrame
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
        lower_bound = features[f'{col}_q25'] - 1.5 * features[f'{col}_iqr']
        upper_bound = features[f'{col}_q75'] + 1.5 * features[f'{col}_iqr']
        features[f'{col}_num_outliers'] = np.sum((data < lower_bound) | (data > upper_bound))

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

# Función para procesar un solo archivo FCS y extraer características
def process_single_fcs_file(fcs_path: str) -> pd.Series:
    sample = FCMeasurement(ID='Sample', datafile=fcs_path)
    df = sample.data
    return extract_distribution_features(df)

# Función principal para procesar todos los archivos en una carpeta
def main():
    # folder_path = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs'
    # output_folder = os.path.join(folder_path, 'Variables')
    # Cambiar a la ruta según conveniencia
    folder_path = r'C:\Users\Alex\Desktop\Maestría Ingeniería Biomédica\TFM\Código\FCSFILES\fcs\Archivos FCS para test externo'
    output_folder = os.path.join(folder_path, 'Variables test externo')    

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    fcs_files = [f for f in os.listdir(folder_path) if f.endswith('.fcs')]
    if not fcs_files:
        print("No se encontraron archivos .fcs en la carpeta.")
        return

    print(f"Se encontraron {len(fcs_files)} archivos .fcs. Procesando...")

    for fcs_filename in fcs_files:              
        input_name = os.path.splitext(fcs_filename)[0]
        output_path = os.path.join(output_folder, f"{input_name}_features.csv")

        if os.path.isfile(output_path):
            print(f"⏩ Ya procesado: {input_name}")
            continue

        fcs_path = os.path.join(folder_path, fcs_filename)
        start_time = time.time()
        print(f"\n🔄 Procesando archivo: {fcs_filename}")

        try:
            features = process_single_fcs_file(fcs_path)
            features.to_csv(output_path)

            elapsed_time = time.time() - start_time
            print(f"✔ Procesado: {input_name} en {elapsed_time:.2f} segundos")
        except Exception as e:
            print(f"❌ Error procesando '{fcs_filename}': {e}")

    start_time = time.time()
    print(f"\nProcesando archivo: {fcs_filename}")
    
    features = process_single_fcs_file(fcs_path)

    output_path = os.path.join(output_folder, f"{input_name}_features.csv")
    features.to_csv(output_path)

    elapsed_time = time.time() - start_time
    print(f"\n✔ Procesamiento completado:")
    print(f"- Features generadas: {len(features)}")
    print(f"- Tiempo total: {elapsed_time:.2f} segundos")
    print(f"- Archivo guardado en: {output_path}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
