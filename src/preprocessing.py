from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.stats import chi2_contingency, chi2
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

def get_spark_session(app_name="LendingClub_EDA"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.bindAddress", "127.0.0.1") \
        .getOrCreate()

def load_raw_data(spark, path):
    # 9.10.4.1: Cargar el dataset CSV
    return spark.read.csv(path, header=True, inferSchema=True)

def create_target(df):
    """
    9.10.3: Generar la variable binaria 'default'
    0 = Fully Paid, 1 = Charged Off
    """
    # Filtramos para quedarnos solo con las clases de interés
    df_filtered = df.filter(F.col("loan_status").isin("Fully Paid", "Charged Off"))
    
    # Aplicamos la lógica del profesor
    df_with_target = df_filtered.withColumn(
        "default", 
        F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0)
    )
    return df_with_target


def run_categorical_association(df, target_col='default'):
    """
    Realiza pruebas de Chi-cuadrado y tamaño del efecto para variables categóricas.
    """
    string_cols = [c for c, dtype in df.dtypes if dtype == 'string' and c != target_col]
    results = []
    n_total = df.count()

    for col_name in string_cols:
        # Generar tabla de contingencia en Spark (eficiente)
        ct = df.crosstab(col_name, target_col).toPandas()
        ct_array = ct.iloc[:, 1:].values # Extraer solo los conteos
        
        # Prueba Chi-cuadrado
        stat, p, dof, expected = chi2_contingency(ct_array)
        
        # Valor crítico (alpha = 0.05)
        critical_val = chi2.ppf(0.95, dof)
        
        # Tamaño del efecto (V de Cramér)
        # V = sqrt(chi2 / (n * (min(cols, rows) - 1)))
        k = min(ct_array.shape)
        cramer_v = np.sqrt(stat / (n_total * (k - 1)))
        
        results.append({
            'variable': col_name,
            'chi2_stat': stat,
            'df': dof,
            'critical_value': critical_val,
            'p_value': p,
            'cramer_v': cramer_v
        })

    # Crear DataFrame de resultados
    res_df = pd.DataFrame(results)

    # Ajuste de Holm para comparaciones múltiples
    _, p_adj, _, _ = multipletests(res_df['p_value'], method='holm')
    res_df['p_value_holm'] = p_adj

    # Estrellas de significancia
    res_df['significance'] = res_df['p_value_holm'].apply(
        lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns'))
    )

    # Interpretación del efecto (Cohen, 1988)
    def interpret_v(v):
        if v < 0.1: return 'Negligible'
        if v < 0.3: return 'Small'
        if v < 0.5: return 'Medium'
        return 'Large'
    
    res_df['effect_interpretation'] = res_df['cramer_v'].apply(interpret_v)
    
    return res_df.sort_values(by='cramer_v', ascending=False)