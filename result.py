import pandas as pd

# Cargar el archivo CSV en un DataFrame de pandas
data = pd.read_csv("result.csv", delimiter=";")

# Calcular la potencia en Watts para todas las columnas de energía
energy_columns = ['package_0', 'dram_0', 'core_0', 'uncore_0']  # Agrega aquí todas las columnas de energía que desees calcular

for column in energy_columns:
    power_column_name = column + '_power_watts'
    data[power_column_name] = (data[column] / 1e6) / data['duration']

# Mostrar las columnas de potencia resultantes
print(data[['timestamp', 'tag'] + [column + '_power_watts' for column in energy_columns]])
