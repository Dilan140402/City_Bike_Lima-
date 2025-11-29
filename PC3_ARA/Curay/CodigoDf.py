import pandas as pd
import numpy as np

# Configuración de Archivos
INPUT_CSV = "citybike_lima.csv"       # Tu archivo crudo original
OUTPUT_CSV = "citybike_procesado.csv" # El archivo limpio resultante

# --- FUNCIONES DE APOYO ---
def periodo_de_dia(h):
    try:
        h = int(h)
    except Exception:
        return np.nan
    if 5 <= h < 12: return "mañana"
    if 12 <= h < 18: return "tarde" # Ajusté ligeramente para separar noche
    if 18 <= h < 22: return "noche"
    return "madrugada"

def safe_mode(series):
    s = series.dropna()
    if s.empty: return ""
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else s.iloc[0]

# --- 1) LECTURA Y RENOMBRE DE COLUMNAS ---
print(">>> Leyendo archivo CSV...")
df = pd.read_csv(INPUT_CSV)

# Mapa de alias para normalizar nombres
alias_map = {
    'id_estacion': ['id_estacion', 'station_id', 'stationid'],
    'nombre_estacion': ['nombre_estacion', 'station_name', 'name'],
    'latitud': ['lat', 'latitude', 'latitud'],
    'longitud': ['lon', 'lng', 'longitude'],
    'capacidad': ['capacidad', 'capacity', 'dockcount'],
    'bicis_libres': ['bicis_libres', 'free_bikes', 'available_bikes'],
    'espacios_vacios': ['espacios_vacios', 'empty_slots', 'empty_docks'],
    'timestamp': ['timestamp', 'scrape_timestamp', 'date'],
    'temp_c': ['temp_c', 'temp_C', 'temp', 'temperature'],
    'vel_viento': ['vel_viento', 'wind_speed'],
    'en_miraflores': ['en_miraflores', 'in_miraflores']
}

col_lower = {c.lower(): c for c in df.columns}
for target, aliases in alias_map.items():
    if target in df.columns: continue
    for a in aliases:
        if a.lower() in col_lower:
            df.rename(columns={col_lower[a.lower()]: target}, inplace=True)
            col_lower = {c.lower(): c for c in df.columns} # Actualizar mapa
            break

# --- 2) CONVERSIÓN DE TIPOS Y LIMPIEZA INICIAL ---
# Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Ordenar por estación y tiempo (CRÍTICO para interpolar temperatura correctamente)
if 'id_estacion' in df.columns:
    df.sort_values(by=['id_estacion', 'timestamp'], inplace=True)

# Columnas numéricas
cols_num = ['latitud', 'longitud', 'capacidad', 'bicis_libres', 'temp_c', 'vel_viento']
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# --- 3) MEJORA: RELLENO INTELIGENTE DE TEMPERATURA ---
if 'temp_c' in df.columns:
    nulos_inicio = df['temp_c'].isnull().sum()
    # Interpolación lineal (rellena huecos basándose en el valor anterior y siguiente en el tiempo)
    df['temp_c'] = df['temp_c'].interpolate(method='linear', limit_direction='both')
    
    # Si quedan nulos (ej. al principio de todo), rellenar con promedio por hora
    if df['temp_c'].isnull().sum() > 0:
        df['hora_temp'] = df['timestamp'].dt.hour
        df['temp_c'] = df['temp_c'].fillna(df.groupby('hora_temp')['temp_c'].transform('mean'))
        df.drop(columns=['hora_temp'], inplace=True)
    
    # Relleno final de seguridad con promedio global (si falló todo lo anterior)
    df['temp_c'] = df['temp_c'].fillna(df['temp_c'].mean())
    print(f"Temperatura corregida. Nulos antes: {nulos_inicio}, Nulos ahora: {df['temp_c'].isnull().sum()}")

# Eliminar columna vacía de viento
if 'vel_viento' in df.columns:
    df.drop(columns=['vel_viento'], inplace=True)

# --- 4) CÁLCULOS DERIVADOS ---
# Espacios vacíos
if 'espacios_vacios' not in df.columns and 'capacidad' in df.columns:
    df['espacios_vacios'] = (df['capacidad'] - df['bicis_libres']).clip(lower=0)

# Ocupación (La variable clave para tus modelos)
# Recalculamos siempre para asegurar precisión matemática
if 'capacidad' in df.columns and 'bicis_libres' in df.columns:
    # Evitar división por cero
    df = df[df['capacidad'] > 0].copy()
    # Fórmula: Bicis Libres / Capacidad Total (0=Vacía, 1=Llena de bicis)
    # Nota: A veces 'ocupación' se define como % de espacios ocupados. 
    # Aquí seguimos tu lógica: Tasa de bicis disponibles.
    df['ocupacion'] = df['bicis_libres'] / df['capacidad']
    df['ocupacion'] = df['ocupacion'].clip(0, 1)

# Fechas
df['fecha'] = df['timestamp'].dt.date
df['hora'] = df['timestamp'].dt.hour
df['dia_semana'] = df['timestamp'].dt.day_name()
df['periodo_dia'] = df['hora'].apply(lambda x: periodo_de_dia(x))

# Códigos de estación (Limpieza específica)
if 'nombre_estacion' in df.columns and 'codigo_estacion' not in df.columns:
    df['codigo_estacion'] = df['nombre_estacion'].astype(str).str.extract(r'(\d{5})')

# Eliminar estación basura '27042' si existe
if 'codigo_estacion' in df.columns:
    df = df[~df['codigo_estacion'].astype(str).str.contains('27042', na=False)]

# --- 5) AGREGADOS Y CATEGORÍAS (TUS REGLAS DE NEGOCIO) ---
# Definir categorías de ocupación
bins = [-0.1, 0.35, 0.65, 1.1] # Ajustado para incluir 0 y 1
labels = ['Baja', 'Media', 'Alta']
df['categoria_ocupacion'] = pd.cut(df['ocupacion'], bins=bins, labels=labels)

# Calcular promedios históricos por estación
cols_agg = {
    'capacidad': 'mean',
    'bicis_libres': 'mean',
    'ocupacion': 'mean'
}
# Agrupamos por ID
station_stats = df.groupby('id_estacion').agg(cols_agg).reset_index()
station_stats.columns = ['id_estacion', 'capacidad_promedio', 'bicis_promedio', 'ocupacion_promedio']

# Categoría promedio
station_stats['categoria_ocupacion_promedio'] = pd.cut(station_stats['ocupacion_promedio'], bins=bins, labels=labels)

# Unir al DF principal
df = df.merge(station_stats, on='id_estacion', how='left')

# --- 6) GUARDADO FINAL ---
# Seleccionar y ordenar columnas relevantes
cols_final = [
    'timestamp', 'fecha', 'hora', 'dia_semana', 'periodo_dia',
    'id_estacion', 'codigo_estacion', 'nombre_estacion',
    'latitud', 'longitud', 'en_miraflores',
    'capacidad', 'capacidad_promedio',
    'bicis_libres', 'bicis_promedio',
    'espacios_vacios',
    'ocupacion', 'ocupacion_promedio',
    'categoria_ocupacion', 'categoria_ocupacion_promedio',
    'temp_c'
]
# Filtrar solo las que existen
cols_final = [c for c in cols_final if c in df.columns]

df[cols_final].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"✅ Procesamiento completado con éxito.")
print(f"   Archivo generado: {OUTPUT_CSV}")
print(f"   Registros totales: {len(df)}")
print(f"   Columnas clave: ocupacion, categoria_ocupacion, temp_c")