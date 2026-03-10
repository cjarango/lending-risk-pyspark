import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

# --- MÓDULO DE FORMATO Y PRESENTACIÓN ---

def format_axis_thousands(x, pos):
    """
    Formateador para ejes de gráficas. Escala valores a 'K' para miles 
    si el valor es >= 1000, manteniendo la legibilidad.
    """
    if x >= 1000:
        return f'{x/1000:,.0f}K'
    return f'{x:,.0f}'

def safe_format_precision(x):
    """
    Aplica formato con separador de miles y 2 decimales. 
    Diseñado para reportes financieros y tablas de métricas.
    """
    if isinstance(x, (int, float, np.number)) and pd.notnull(x):
        return "{:,.2f}".format(x)
    return x

def safe_format_integer(x):
    """
    Aplica formato con separador de miles sin decimales. 
    Ideal para conteos de registros y auditoría de volúmenes.
    """
    if isinstance(x, (int, float, np.number)) and pd.notnull(x):
        return "{:,.0f}".format(x)
    return x

# --- MÓDULO DE TRANSFORMACIÓN DE DATOS (SPARK) ---

def cast_columns_to_double(df, cols_to_cast):
    """
    Convierte de forma masiva las columnas especificadas al tipo DoubleType.
    Garantiza la compatibilidad con los algoritmos de Spark MLlib.
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame de Spark.
        cols_to_cast (list): Lista de nombres de columnas a transformar.
    """
    for col_name in cols_to_cast:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    return df

# --- MÓDULO DE EVALUACIÓN DE MODELOS ---

def calculate_metrics_from_cm(conf_matrix):
    """
    Deriva métricas de clasificación a partir de una matriz de confusión 2x2.
    
    Args:
        conf_matrix (list/ndarray): Matriz en formato [[TN, FP], [FN, TP]].
        
    Returns:
        dict: Diccionario con Accuracy, Precision, Recall y F1-Score.
    """
    # Desglose de componentes de la matriz
    tn, fp = conf_matrix[0]
    fn, tp = conf_matrix[1]
    
    # Cálculos aritméticos
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1-Score (Media armónica)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1
    }