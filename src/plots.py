import io
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm, median_abs_deviation
from sklearn.metrics import roc_curve, auc
from IPython.display import display, HTML

def plot_boxplot(df, var_col, var_label, target_col, stats_vector, sample_size=20000):
    """
    Genera un diagrama de caja comparativo con métricas de inferencia estadística.
    
    Args:
        df (pyspark.sql.DataFrame): Conjunto de datos de entrada.
        var_col (str): Nombre de la variable numérica a analizar.
        var_label (str): Etiqueta descriptiva para los ejes y título.
        target_col (str): Variable objetivo para la segmentación.
        stats_vector (list): Vector con [p-valor, estadístico r].
        sample_size (int): Tamaño de la muestra para la visualización.
    """
    # Extracción y muestreo de datos
    count = df.count()
    fraction = min(1.0, (sample_size * 1.2) / count)
    pdf = df.select(var_col, target_col).sample(False, fraction, seed=42).limit(sample_size).toPandas()
    
    # Mapeo de categorías
    pdf['Grupo'] = pdf[target_col].map({0: 'Cumplido (0)', 1: 'Default (1)'})
    
    # Procesamiento de significancia y efecto
    p_val, r_val = stats_vector[0], stats_vector[1]
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    def _get_effect_magnitude(r):
        if r < 0.1: return 'Negligible'
        if r < 0.3: return 'Small'
        if r < 0.5: return 'Medium'
        return 'Large'
    
    # Configuración de estilo editorial
    sns.set_style("white")
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})
    
    fig, ax = plt.subplots(figsize=(7, 5)) 
    
    sns.boxplot(
        x='Grupo', 
        y=var_col, 
        data=pdf, 
        hue='Grupo',
        palette=["#95a5a6", "#c0392b"], 
        legend=False, 
        showfliers=False, 
        width=0.4,
        ax=ax
    )
    
    # Estructura visual del marco
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
        spine.set_visible(True)
    
    # Anotación de resultados estadísticos
    p_display = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.4f}"
    stats_text = (
        f"Test de Wilcoxon\n"
        f"p-adj: {p_display} ({sig})\n"
        f"Efecto r: {r_val:.3f} ({_get_effect_magnitude(r_val)})"
    )

    at = AnchoredText(
        stats_text, loc=1, prop=dict(size=10, family='serif'), 
        frameon=True, bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes
    )
    at.patch.set_boxstyle("square,pad=0.3")
    at.patch.set_edgecolor('black')
    ax.add_artist(at)

    ax.set_title(f"Distribución de {var_label} por Estado del Crédito", fontsize=12, pad=15, weight='bold')
    ax.set_xlabel("Estado Final del Préstamo", fontsize=11, weight='bold')
    ax.set_ylabel(var_label, fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_dual_lime(idx_fn, idx_tn, title_fn, title_tn, explainer, X_test_imp, model):
    """
    Visualiza explicaciones LIME para casos de Falso Negativo y Verdadero Negativo.
    
    Args:
        idx_fn (int): Índice del caso Falso Negativo.
        idx_tn (int): Índice del caso Verdadero Negativo.
        title_fn (str): Subtítulo para el panel de Falso Negativo.
        title_tn (str): Subtítulo para el panel de Verdadero Negativo.
        explainer: Instancia del explicador LIME.
        X_test_imp (ndarray): Matriz de características de prueba.
        model: Modelo predictivo entrenado.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    sns.set_style("white")
    
    exp_fn = explainer.explain_instance(X_test_imp[idx_fn], model.predict_proba, num_features=8)
    exp_tn = explainer.explain_instance(X_test_imp[idx_tn], model.predict_proba, num_features=8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'wspace': 0.65})
    
    def _draw_panel(exp, ax, title):
        list_exp = exp.as_list()
        features = [x[0] for x in list_exp]
        weights = [x[1] for x in list_exp]
        features.reverse()
        weights.reverse()
        
        colors = ['#c0392b' if w > 0 else '#95a5a6' for w in weights]
        
        ax.barh(features, weights, color=colors, edgecolor='black', linewidth=0.8)
        ax.axvline(0, color='black', lw=1, zorder=3)
        ax.set_title(title, fontsize=12, weight='bold', pad=15)
        ax.set_xlabel('Peso (Impacto en la Decisión)', fontsize=10, weight='bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
            
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)

    _draw_panel(exp_fn, ax1, f'Falso Negativo: {title_fn}')
    _draw_panel(exp_tn, ax2, f'Verdadero Negativo: {title_tn}')

    plt.tight_layout()
    plt.show()

def plot_roc_comparison(y_test_sk, y_prob_sk, spark_predictions):
    """
    Compara las curvas ROC de las implementaciones Scikit-Learn y PySpark.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    sns.set_style("white")
    
    # Curva Scikit-Learn
    fpr_sk, tpr_sk, _ = roc_curve(y_test_sk, y_prob_sk)
    roc_auc_sk = auc(fpr_sk, tpr_sk)
    
    # Curva PySpark
    probs_spark = spark_predictions.select("probability", "default").collect()
    y_prob_sp = [row["probability"][1] for row in probs_spark]
    y_test_sp = [row["default"] for row in probs_spark]
    fpr_sp, tpr_sp, _ = roc_curve(y_test_sp, y_prob_sp)
    roc_auc_sp = auc(fpr_sp, tpr_sp)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(fpr_sk, tpr_sk, color="#2c3e50", lw=2, label=f'Scikit-Learn (AUC = {roc_auc_sk:.4f})')
    ax.plot(fpr_sp, tpr_sp, color="#c0392b", lw=2, linestyle='--', label=f'PySpark (AUC = {roc_auc_sp:.4f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=1, linestyle=':')

    ax.set_title('Comparativa de Rendimiento: Curvas ROC', fontsize=14, pad=20, weight='bold')
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=11, fontweight='bold')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(frameon=False, loc='lower right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_full_diagnostic(df, var, var_label, sample_n=5000, figsize=(8, 3.5)):
    """
    Genera un panel de diagnóstico morfológico (histograma y boxplot) centrado.
    """
    # Procesamiento robusto de datos
    count = df.count()
    data = df.select(var).dropna().sample(False, min(1.0, sample_n/count), seed=42).toPandas()[var]
    
    mu, std = data.mean(), data.std()
    median = data.median()
    mad_std = median_abs_deviation(data, scale='normal') 
    
    lower_hampel = median - 3 * mad_std
    upper_hampel = median + 3 * mad_std

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histograma y curva de normalidad teórica
    sns.histplot(data, kde=False, stat="density", color="#95a5a6", alpha=0.6, 
                 edgecolor='black', linewidth=1, ax=axes[0])
    x_range = np.linspace(data.min(), data.max(), 200)
    axes[0].plot(x_range, norm.pdf(x_range, mu, std), color="#c0392b", lw=2, linestyle="--")
    
    axes[0].set_xlabel(var_label, fontsize=10, fontweight='bold', labelpad=10)
    axes[0].set_ylabel('Densidad', fontsize=10, fontweight='bold', labelpad=10)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)

    # Boxplot con identificadores de filtro Hampel
    sns.boxplot(x=data, ax=axes[1], color="#95a5a6", fliersize=3, linewidth=1.2, width=0.4)
    axes[1].axvline(lower_hampel, color="#c0392b", linestyle=":", lw=2.5)
    axes[1].axvline(upper_hampel, color="#c0392b", linestyle=":", lw=2.5)
    axes[1].set_xlabel(var_label, fontsize=10, fontweight='bold', labelpad=10)
    axes[1].set_yticks([])

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    
    # Exportación a HTML centrado
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    display(HTML(f"""
        <div style="text-align: center; width: 100%;">
            <img src="data:image/png;base64,{data_uri}" style="display: inline-block;">
        </div>
    """))