import numpy as np
import pandas as pd
from typing import List

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer as SparkImputer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as RF_Spark


def _calculate_f1_from_cm(cm: np.ndarray) -> float:
    """
    [MÉTODO PRIVADO] Calcula la media armónica (F1-Score) a partir de una matriz de confusión.
    
    Args:
        cm (np.ndarray): Matriz de confusión binaria en formato scikit-learn [[TN, FP], [FN, TP]].
        
    Returns:
        float: Valor del F1-Score (entre 0.0 y 1.0).
    """
    tn, fp, fn, tp = cm.ravel()
    
    # Tolerancia a divisiones por cero en pliegues sesgados
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def manual_grid_search_rf(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_estimators_list: List[int], 
    max_depth_list: List[int], 
    n_splits: int = 3, 
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Ejecuta un Grid Search con validación cruzada estratificada para Random Forest.
    Integra control estricto de Data Leakage (imputación intra-pliegue) y soporte dual CPU/GPU.
    
    Args:
        X (pd.DataFrame): Matriz de características (entrenamiento).
        y (pd.Series): Vector objetivo binario (default).
        n_estimators_list (List[int]): Cuadrícula de hiperparámetros para el número de árboles.
        max_depth_list (List[int]): Cuadrícula de hiperparámetros para la profundidad máxima.
        n_splits (int, opcional): Número de pliegues para CV. Por defecto es 3.
        use_gpu (bool, opcional): Si es True, intenta instanciar cuML (NVIDIA) en lugar de scikit-learn.
        
    Returns:
        pd.DataFrame: Tabla resumen con el F1-Score promedio para cada combinación de hiperparámetros.
    """
    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 1. Resolución dinámica del estimador (Hardware Bridge)
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestClassifier
            print("Iniciando motor CUDA (cuML RandomForestClassifier)...")
        except ImportError:
            print("Advertencia: cuML no detectado. Revirtiendo a CPU.")
            from sklearn.ensemble import RandomForestClassifier
            use_gpu = False
    else:
        from sklearn.ensemble import RandomForestClassifier
        print("Iniciando motor CPU (scikit-learn)...")

    for depth in max_depth_list:
        for n_est in n_estimators_list:
            f1_folds = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 2. Imputación sin fuga de datos
                imp = SimpleImputer(strategy='median')
                X_tr_imp = imp.fit_transform(X_tr)
                X_val_imp = imp.transform(X_val)
                
                # 3. Casteo de memoria para aceleración de tensores en GPU
                if use_gpu:
                    X_tr_imp = np.float32(X_tr_imp)
                    X_val_imp = np.float32(X_val_imp)
                    y_tr = np.int32(y_tr)
                
                # 4. Instanciación y Entrenamiento
                rf_kwargs = {'n_estimators': n_est, 'max_depth': depth, 'random_state': 42}
                if not use_gpu:
                    rf_kwargs['n_jobs'] = -1
                    rf_kwargs['class_weight'] = 'balanced' 
                
                rf = RandomForestClassifier(**rf_kwargs)
                rf.fit(X_tr_imp, y_tr)
                
                # 5. Evaluación mediante matriz de confusión
                preds = rf.predict(X_val_imp)
                cm = confusion_matrix(y_val, preds)
                f1_folds.append(_calculate_f1_from_cm(cm))
            
            avg_f1 = np.mean(f1_folds)
            results.append({'max_depth': depth, 'n_estimators': n_est, 'f1_score': avg_f1})
            
    return pd.DataFrame(results)


def manual_grid_search_pyspark(
    df_raw: SparkDataFrame, 
    num_cols: List[str], 
    n_estimators_list: List[int], 
    max_depth_list: List[int], 
    n_splits: int = 5, 
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Ejecuta un Grid Search distribuido nativo en PySpark para Random Forest.
    Aplica imputación controlada por pliegue y optimización del DAG de Spark (single-pass eval).
    
    Args:
        df_raw (SparkDataFrame): Dataset particionado distribuido.
        num_cols (List[str]): Nombres de las columnas numéricas que requieren imputación.
        n_estimators_list (List[int]): Opciones de número de árboles.
        max_depth_list (List[int]): Opciones de profundidad máxima.
        n_splits (int, opcional): Número de pliegues para validación cruzada. Por defecto es 5.
        use_gpu (bool, opcional): Activa alertas sobre configuraciones RAPIDS (solo ETL para MLlib).
        
    Returns:
        pd.DataFrame: Tabla resumen (en memoria local) con los resultados del Grid Search.
    """
    results = []
    
    if use_gpu:
        print("Nota de Hardware: Activando optimización de ETL en GPU (RAPIDS).")
        print("Advertencia: RandomForest de MLlib entrena en CPU por defecto.")
    
    # Detección automática de columnas categóricas indexadas/codificadas
    cols_idx_ohe = [c for c in df_raw.columns if c.endswith('_idx') or c.endswith('_ohe')]
    df_folds = df_raw.withColumn("fold", (F.rand(seed=42) * n_splits).cast("int"))

    print("Iniciando Grid Search distribuido con F1 Manual...")

    for depth in max_depth_list:
        for n_est in n_estimators_list:
            f1_folds = []
            
            for i in range(n_splits):
                train_raw = df_folds.filter(F.col("fold") != i)
                val_raw = df_folds.filter(F.col("fold") == i)
                
                # --- A. IMPUTACIÓN AISLADA ---
                imputer = SparkImputer(
                    strategy="median",
                    inputCols=num_cols,
                    outputCols=[f"{c}_imp" for c in num_cols]
                )
                imputer_model = imputer.fit(train_raw)
                train_imp = imputer_model.transform(train_raw)
                val_imp = imputer_model.transform(val_raw)
                
                # --- B. ENSAMBLADO VECTORIAL ---
                cols_num_imp = [f"{c}_imp" for c in num_cols]
                assembler = VectorAssembler(
                    inputCols=cols_num_imp + cols_idx_ohe, 
                    outputCol="features_fold",
                    handleInvalid="keep"
                )
                train_final = assembler.transform(train_imp).select("features_fold", "default")
                val_final = assembler.transform(val_imp).select("features_fold", "default")
                
                # --- C. BALANCEO DE CLASES (Pesos Dinámicos) ---
                stats = train_final.groupBy("default").count().collect()
                counts = {row['default']: row['count'] for row in stats}
                total_n = sum(counts.values())
                w_pos = total_n / (2.0 * counts.get(1, 1))
                w_neg = total_n / (2.0 * counts.get(0, 1))
                
                train_weighted = train_final.withColumn("weight_col", 
                    F.when(F.col("default") == 1, w_pos).otherwise(w_neg))
                
                # --- D. ENTRENAMIENTO RF_SPARK ---
                rf_spark = RF_Spark(
                    labelCol="default", 
                    featuresCol="features_fold", 
                    weightCol="weight_col",
                    numTrees=n_est, 
                    maxDepth=depth,
                    maxBins=64,
                    seed=42
                )
                model = rf_spark.fit(train_weighted)
                
                # --- E. INTEGRACIÓN DE MÉTRICA MANUAL (Optimización de DAG) ---
                predictions = model.transform(val_final)
                
                # Extracción optimizada (Un solo Action por el clúster)
                conf_matrix_spark = predictions.groupBy("prediction", "default").count().collect()
                
                tn, fp, fn, tp = 0, 0, 0, 0
                for row in conf_matrix_spark:
                    pred, actual, count = row['prediction'], row['default'], row['count']
                    if pred == 1.0 and actual == 1:
                        tp = count
                    elif pred == 1.0 and actual == 0:
                        fp = count
                    elif pred == 0.0 and actual == 1:
                        fn = count
                    elif pred == 0.0 and actual == 0:
                        tn = count
                
                # Puente hacia la función privada unificada
                cm = np.array([[tn, fp], [fn, tp]])
                f1_folds.append(_calculate_f1_from_cm(cm))
            
            avg_f1 = np.mean(f1_folds)
            results.append({'max_depth': depth, 'n_estimators': n_est, 'f1_score': avg_f1})
            print(f"   Trees={n_est}, Depth={depth} | F1 Real (Clase 1): {avg_f1:.4f}")
            
    return pd.DataFrame(results)