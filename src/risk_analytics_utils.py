import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (shapiro, jarque_bera, kstest, skew, kurtosis, 
                         median_abs_deviation, chi2_contingency, chi2, 
                         ks_2samp, spearmanr, kendalltau, mannwhitneyu, levene)
from statsmodels.stats.multitest import multipletests
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# --- UTILIDADES INTERNAS ---

def _format_p_value(p):
    """Aplica formato científico y estrellas de significancia estadística."""
    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ' (ns)'))
    if p < 0.001:
        return f"< 0.001{stars}"
    return f"{p:.4f}{stars}"

def _interpret_effect_size(val):
    """Clasifica la magnitud del efecto según umbrales estándar."""
    if val < 0.1: return 'Despreciable'
    if val < 0.3: return 'Pequeño'
    if val < 0.5: return 'Mediano'
    return 'Grande'

# --- MÓDULO DE CORRELACIÓN Y ESTRUCTURA ---

def get_correlation_matrix(df, feature_list):
    """
    Calcula la matriz de correlación de Spearman en un entorno distribuido.
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame de Spark.
        feature_list (list): Lista de columnas numéricas.
        
    Returns:
        pd.DataFrame: Matriz de correlación simétrica.
    """
    df_temp = df.select(feature_list).fillna(0)
    assembler = VectorAssembler(inputCols=feature_list, outputCol="corr_features")
    df_vectorized = assembler.transform(df_temp).select("corr_features")
    
    matrix_obj = Correlation.corr(df_vectorized, "corr_features", method="spearman").collect()[0][0]
    
    return pd.DataFrame(
        matrix_obj.toArray(), 
        index=feature_list, 
        columns=feature_list
    )

# --- MÓDULO DE DISTRIBUCIÓN Y ANOMALÍAS ---

def validate_outliers_robust(df, var, threshold=3.5, sample_n=100000):
    """
    Identifica valores atípicos mediante el estadístico de Hampel (Z-Score modificado).
    Utiliza la fórmula: $$M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}$$
    """
    fraction = min(1.0, sample_n / df.count())
    pdf = df.select(var).dropna().sample(False, fraction, seed=42).toPandas()
    data = pdf[var]
    
    median = data.median()
    mad_raw = median_abs_deviation(data, scale=1.0) 
    
    if mad_raw == 0:
        return pd.DataFrame()
    
    pdf['modified_z'] = 0.6745 * (data - median) / mad_raw
    pdf['abs_modified_z'] = pdf['modified_z'].abs()
    
    outliers_df = pdf[pdf['abs_modified_z'] > threshold].copy()
    
    if outliers_df.empty:
        return pd.DataFrame()

    return outliers_df.sort_values(by='abs_modified_z', ascending=False).head(10)[[var, 'modified_z']].reset_index(drop=True)

def run_normality_tests(df, variables, sample_n=5000):
    """
    Ejecuta una batería de pruebas de normalidad (Shapiro-Wilk, Jarque-Bera, K-S).
    """
    fraction = min(1.0, sample_n / df.count())
    pdf_sample = df.select(variables).dropna().sample(False, fraction, seed=42).toPandas()
    
    results = []
    for var in variables:
        data = pdf_sample[var]
        
        # Pruebas estadísticas
        stat_sw, p_sw = shapiro(data)
        stat_jb, p_jb = jarque_bera(data)
        data_std = (data - data.mean()) / data.std()
        stat_ks, p_ks = kstest(data_std, 'norm')
        
        results.append({
            'Atributo': var,
            'Asimetría': round(skew(data), 3),
            'Curtosis': round(kurtosis(data), 3),
            'Shapiro-Wilk': f"{stat_sw:.3f}{_format_p_value(p_sw).split(' ')[-1]}",
            'Jarque-Bera': f"{stat_jb:.1f}{_format_p_value(p_jb).split(' ')[-1]}",
            'K-S (Lilliefors)': f"{stat_ks:.3f}{_format_p_value(p_ks).split(' ')[-1]}",
            '¿Normal?': 'SÍ' if (p_sw > 0.05 and p_jb > 0.05 and p_ks > 0.05) else 'NO'
        })

    return pd.DataFrame(results).sort_values(by='Asimetría', ascending=False)

# --- MÓDULO DE INFERENCIA Y ASOCIACIÓN ---

def robust_contrast(df, feature_list, target_col='default', sample_size=50000):
    """
    Contraste de hipótesis robusto para variables numéricas vs categóricas.
    Calcula el tamaño del efecto r de Rosenthal: $$r = \frac{|Z|}{\sqrt{N}}$$
    """
    fraction = min(1.0, (sample_size * 1.5) / df.count())
    pdf = df.select(feature_list + [target_col]).sample(False, fraction, seed=42).limit(sample_size).toPandas()
    
    results = []
    g0 = pdf[pdf[target_col] == 0]
    g1 = pdf[pdf[target_col] == 1]

    for col in feature_list:
        v0, v1 = g0[col].dropna(), g1[col].dropna()
        if len(v0) < 2 or len(v1) < 2: continue

        # Estadísticos descriptivos
        def _get_summary(data):
            q1, me, q3 = np.percentile(data, [25, 50, 75])
            return f"{me:,.2f} ({q3-q1:,.2f})"

        # Tests
        _, p_bf = levene(v0, v1, center='median')
        u_stat, p_wil = mannwhitneyu(v0, v1, alternative='two-sided')
        
        # Efecto Rosenthal
        mu_u = (len(v0) * len(v1)) / 2
        std_u = np.sqrt((len(v0) * len(v1) * (len(v0) + len(v1) + 1)) / 12)
        z_stat = (u_stat - mu_u) / std_u
        effect_r = abs(z_stat) / np.sqrt(len(v0) + len(v1))

        results.append({
            'Variable': col,
            'Me (IQR) No-Def': _get_summary(v0),
            'Me (IQR) Def': _get_summary(v1),
            'Homocedasticidad': 'Sí' if p_bf > 0.05 else 'No',
            'p_val_raw': p_wil,
            'r de Rosenthal': effect_r
        })

    if not results: return pd.DataFrame()
    res_df = pd.DataFrame(results)
    _, p_adj, _, _ = multipletests(res_df['p_val_raw'], method='holm')
    
    res_df['p-valor (Holm)'] = [_format_p_value(p) for p in p_adj]
    res_df['Interpretación'] = res_df['r de Rosenthal'].apply(_interpret_effect_size)
    
    return res_df[['Variable', 'Me (IQR) No-Def', 'Me (IQR) Def', 'Homocedasticidad', 
                   'p-valor (Holm)', 'r de Rosenthal', 'Interpretación']].sort_values(by='r de Rosenthal', ascending=False)

def categorical_association(df, target_col='default'):
    """
    Analiza la asociación entre variables categóricas mediante Chi-cuadrado y V de Cramér.
    $$V = \sqrt{\frac{\chi^2}{n(k-1)}}$$
    """
    string_cols = [c for c, dtype in df.dtypes if dtype == 'string' and c != target_col]
    results = []

    for col_name in string_cols:
        valid_df = df.select(col_name, target_col).dropna()
        n_valid = valid_df.count()
        if valid_df.select(col_name).distinct().count() < 2: continue

        ct = valid_df.crosstab(col_name, target_col).toPandas()
        ct_array = ct.iloc[:, 1:].values
        
        stat, p, dof, _ = chi2_contingency(ct_array)
        critical_val = chi2.ppf(0.95, dof)
        k = min(ct_array.shape)
        cramer_v = np.sqrt(stat / (n_valid * (k - 1))) if k > 1 and n_valid > 0 else 0.0

        results.append({
            'Variable': col_name,
            'Estadístico Chi2': stat,
            'GL': dof,
            'Valor Crítico': critical_val,
            'p_val_raw': p,
            'V de Cramér': cramer_v
        })

    if not results: return pd.DataFrame()
    res_df = pd.DataFrame(results)
    _, p_adj, _, _ = multipletests(res_df['p_val_raw'], method='holm')
    
    res_df['p-valor (Holm)'] = [_format_p_value(p) for p in p_adj]
    res_df['Interpretación'] = res_df['V de Cramér'].apply(_interpret_effect_size)
    
    return res_df[['Variable', 'Estadístico Chi2', 'GL', 'Valor Crítico', 
                   'p-valor (Holm)', 'V de Cramér', 'Interpretación']].sort_values(by='V de Cramér', ascending=False)