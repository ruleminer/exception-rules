import pandas as pd

# Wczytaj plik CSV
df = pd.read_csv('./exp_1_srv/exceptions_summary_ALL.csv')

# Usuń wszystkie wiersze, gdzie model zawiera "algorithm4"
df = df[~df['model'].str.contains('algorithm4')]

# Zamień "algorithm3" na "algorithm" w kolumnie 'model'
df['model'] = df['model'].str.replace('algorithm3', 'algorithm')

# Zapisz zmodyfikowany plik
df.to_csv('./exp_1_srv/exceptions_summary_ALL_cleaned.csv', index=False)

print("Gotowe! Zapisano do exceptions_summary_ALL_cleaned.csv")