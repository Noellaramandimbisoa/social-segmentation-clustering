"""
=============================================================
  PROJET : Social Segmentation & Clustering
  Fichier : social_segmentation_clustering.py
  Description : Analyse complète de segmentation client
                avec clustering K-Means et visualisations
                statiques (Matplotlib/Seaborn) + interactives
                (Plotly) et export Excel (openpyxl).
=============================================================

COLONNES DU CSV (d'après la capture) :
  1. ID       → UUID (identifiant unique)
  2. Age      → int
  3. Gender   → Male / Female
  4. Income   → int (revenu annuel)
  5. Score    → int (score de comportement)

INSTRUCTIONS D'UTILISATION :
  1. Placez votre CSV dans : data/raw/Segmentation Data .csv
  2. Installez les dépendances :  pip install -r requirements.txt
  3. Lancez le script :           python social_segmentation_clustering.py
  4. Les résultats seront dans :  data/processed/  et  results/
=============================================================
"""

# ──────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────
import os
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION (config.yaml ou valeurs par défaut)
# ──────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data": {
        "raw_path":       "data/raw/Segmentation Data .csv",
        "processed_path": "data/processed/segmentation_processed.csv",
        "has_header":     False,
        "columns":        ["ID", "Age", "Gender", "Income", "Score"],
    },
    "clustering": {
        "k_min":       2,
        "k_max":       10,
        "random_state": 42,
        "n_init":      10,
    },
    "output": {
        "results_dir":   "results",
        "excel_report":  "results/rapport_segmentation.xlsx",
        "open_plotly":   False,   # True → ouvre les graphiques Plotly dans le navigateur
    },
}

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"  ✔ config.yaml chargé depuis {path}")
        return cfg
    print("  ℹ  config.yaml introuvable — utilisation des valeurs par défaut")
    return DEFAULT_CONFIG

CFG = load_config()

RAW_PATH       = CFG["data"]["raw_path"]
PROCESSED_PATH = CFG["data"]["processed_path"]
RESULTS_DIR    = CFG["output"]["results_dir"]
EXCEL_PATH     = CFG["output"]["excel_report"]
K_MIN          = CFG["clustering"]["k_min"]
K_MAX          = CFG["clustering"]["k_max"]
RANDOM_STATE   = CFG["clustering"]["random_state"]
N_INIT         = CFG["clustering"]["n_init"]
OPEN_PLOTLY    = CFG["output"]["open_plotly"]

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT ET INSPECTION DES DONNÉES
# ══════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV et affiche un aperçu complet."""
    print("\n" + "="*60)
    print("ÉTAPE 1 : Chargement des données")
    print("="*60)

    df = pd.read_csv(
        path,
        header=None if not CFG["data"]["has_header"] else 0,
        names=CFG["data"]["columns"] if not CFG["data"]["has_header"] else None,
    )

    print(f"  ✔ Fichier chargé      : {path}")
    print(f"  ✔ Dimensions          : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("\n── Aperçu (5 premières lignes) ──")
    print(df.head().to_string())
    print("\n── Types de données ──")
    print(df.dtypes.to_string())
    print("\n── Statistiques descriptives ──")
    print(df.describe(include="all").to_string())

    return df


# ══════════════════════════════════════════════════════════
# ÉTAPE 2 — NETTOYAGE ET PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> tuple:
    """
    Nettoie et normalise les données.

    Retourne :
      df_clean  → DataFrame nettoyé (Gender encodé + Cluster plus tard)
      X_scaled  → matrice numpy normalisée pour clustering
      features  → liste des colonnes numériques utilisées
      scaler    → objet StandardScaler fitté (pour inverse_transform si besoin)
    """
    print("\n" + "="*60)
    print("ÉTAPE 2 : Nettoyage & Prétraitement")
    print("="*60)

    df_clean = df.copy()

    # ── 2a. Valeurs manquantes ──────────────────────────────
    missing = df_clean.isnull().sum()
    print(f"\n  Valeurs manquantes par colonne :\n{missing.to_string()}")
    before = len(df_clean)
    df_clean.dropna(inplace=True)
    print(f"  ✔ Lignes supprimées (NaN) : {before - len(df_clean)}")

    # ── 2b. Doublons ────────────────────────────────────────
    dupes = df_clean.duplicated().sum()
    df_clean.drop_duplicates(inplace=True)
    print(f"  ✔ Doublons supprimés      : {dupes}")

    # ── 2c. Encodage Gender ─────────────────────────────────
    le = LabelEncoder()
    df_clean["Gender_enc"] = le.fit_transform(df_clean["Gender"])
    print(f"  ✔ Encodage Gender         : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 2d. Tests de normalité (SciPy) ──────────────────────
    print("\n  Tests de normalité (Shapiro-Wilk, p-value) :")
    for col in ["Age", "Income", "Score"]:
        sample = df_clean[col].sample(min(500, len(df_clean)), random_state=42)
        _, p = stats.shapiro(sample)
        normal = "✔ normale" if p > 0.05 else "✘ non-normale"
        print(f"    {col:<10} p={p:.4f}  → {normal}")

    # ── 2e. Détection des outliers (Z-score, seuil=3) ───────
    print("\n  Outliers détectés (|Z-score| > 3) :")
    features_num = ["Age", "Income", "Score"]
    for col in features_num:
        z = np.abs(stats.zscore(df_clean[col]))
        n_out = (z > 3).sum()
        print(f"    {col:<10} {n_out} outliers")

    # ── 2f. Normalisation StandardScaler ────────────────────
    features = ["Age", "Gender_enc", "Income", "Score"]
    X = df_clean[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n  ✔ Normalisation appliquée (StandardScaler)")
    print(f"  ✔ Features utilisées      : {features}")

    return df_clean, X_scaled, features, scaler


# ══════════════════════════════════════════════════════════
# ÉTAPE 3 — CHOIX DU NOMBRE OPTIMAL DE CLUSTERS
# ══════════════════════════════════════════════════════════
def find_optimal_k(X_scaled: np.ndarray) -> int:
    """
    Méthode Elbow (inertie) + Silhouette Score avec tqdm.
    Génère des graphiques statiques (PNG) ET interactifs (Plotly HTML).
    """
    print("\n" + "="*60)
    print("ÉTAPE 3 : Recherche du k optimal")
    print("="*60)

    k_range    = range(K_MIN, K_MAX + 1)
    inertias   = []
    silhouettes = []

    for k in tqdm(k_range, desc="  Test K-Means", unit="k"):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Résumé texte
    print("\n  k  | Inertie     | Silhouette")
    print("  ---+-------------+------------")
    for k, ine, sil in zip(k_range, inertias, silhouettes):
        print(f"  {k:<3}| {ine:>11.1f} | {sil:.4f}")

    # ── Graphique statique Matplotlib ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, "o-", color="#2ecc71", lw=2)
    axes[0].set_title("Méthode Elbow (Inertie)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertie (WCSS)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "s-", color="#e74c3c", lw=2)
    axes[1].set_title("Score Silhouette", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Score Silhouette")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "elbow_silhouette.png")

    # ── Graphique interactif Plotly ─────────────────────────
    fig_px = make_subplots(rows=1, cols=2,
                           subplot_titles=("Méthode Elbow", "Score Silhouette"))
    fig_px.add_trace(go.Scatter(x=list(k_range), y=inertias,
                                mode="lines+markers", name="Inertie",
                                line=dict(color="#2ecc71", width=3),
                                marker=dict(size=8)), row=1, col=1)
    fig_px.add_trace(go.Scatter(x=list(k_range), y=silhouettes,
                                mode="lines+markers", name="Silhouette",
                                line=dict(color="#e74c3c", width=3),
                                marker=dict(size=8)), row=1, col=2)
    fig_px.update_layout(title="Choix du nombre optimal de clusters",
                         template="plotly_white", height=450)
    save_plotly(fig_px, "elbow_silhouette_interactive.html")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\n  ✔ Meilleur k (Silhouette max) : k = {best_k}")
    return best_k


# ══════════════════════════════════════════════════════════
# ÉTAPE 4 — APPLICATION DU CLUSTERING K-MEANS
# ══════════════════════════════════════════════════════════
def apply_kmeans(df_clean: pd.DataFrame, X_scaled: np.ndarray, k: int) -> pd.DataFrame:
    """Applique K-Means et ajoute les labels au DataFrame."""
    print("\n" + "="*60)
    print(f"ÉTAPE 4 : K-Means (k={k})")
    print("="*60)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    df_clean = df_clean.copy()
    df_clean["Cluster"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df_clean["Cluster"])
    print(f"  ✔ Silhouette Score final : {sil:.4f}")
    print(f"\n  Taille des clusters :")
    print(df_clean["Cluster"].value_counts().sort_index().to_string())

    return df_clean


# ══════════════════════════════════════════════════════════
# ÉTAPE 5 — PROFIL DES CLUSTERS
# ══════════════════════════════════════════════════════════
def analyze_clusters(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques moyennes + tests statistiques par cluster."""
    print("\n" + "="*60)
    print("ÉTAPE 5 : Profiling des clusters")
    print("="*60)

    profile = df_clean.groupby("Cluster").agg(
        Taille       = ("Cluster", "count"),
        Age_moyen    = ("Age", "mean"),
        Age_std      = ("Age", "std"),
        Revenu_moyen = ("Income", "mean"),
        Revenu_std   = ("Income", "std"),
        Score_moyen  = ("Score", "mean"),
        Score_std    = ("Score", "std"),
        Pct_hommes   = ("Gender_enc", lambda x: round(x.mean() * 100, 1)),
    ).round(2)

    print(profile.to_string())

    # ── Test ANOVA entre clusters (SciPy) ───────────────────
    print("\n  Tests ANOVA (différence entre clusters) :")
    for col in ["Age", "Income", "Score"]:
        groups = [df_clean[df_clean["Cluster"] == c][col].values
                  for c in df_clean["Cluster"].unique()]
        f, p = stats.f_oneway(*groups)
        sig = "✔ significatif" if p < 0.05 else "✘ non-significatif"
        print(f"    {col:<10} F={f:.2f}  p={p:.4f}  → {sig}")

    return profile


# ══════════════════════════════════════════════════════════
# ÉTAPE 6 — VISUALISATIONS (Matplotlib + Plotly)
# ══════════════════════════════════════════════════════════
def visualize_clusters(df_clean: pd.DataFrame, X_scaled: np.ndarray):
    """Génère toutes les visualisations statiques ET interactives."""
    print("\n" + "="*60)
    print("ÉTAPE 6 : Visualisations")
    print("="*60)

    k       = df_clean["Cluster"].nunique()
    palette = sns.color_palette("tab10", k)
    px_pal  = px.colors.qualitative.Set1[:k]

    # ── PCA 2D ──────────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var   = pca.explained_variance_ratio_

    df_pca = df_clean.copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    # Statique
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(k):
        m = df_pca["Cluster"] == c
        ax.scatter(df_pca.loc[m, "PC1"], df_pca.loc[m, "PC2"],
                   label=f"Cluster {c}", color=palette[c],
                   alpha=0.7, edgecolors="white", linewidth=0.5, s=60)
    ax.set_title("Clustering — PCA 2D", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.legend(title="Cluster"); ax.grid(alpha=0.2)
    plt.tight_layout(); save_fig(fig, "pca_clusters.png")

    # Interactif Plotly
    fig_px = px.scatter(df_pca, x="PC1", y="PC2",
                        color=df_pca["Cluster"].astype(str),
                        hover_data=["Age", "Gender", "Income", "Score"],
                        title="Clustering — PCA 2D (interactif)",
                        color_discrete_sequence=px_pal,
                        labels={"color": "Cluster"},
                        template="plotly_white")
    fig_px.update_traces(marker=dict(size=7, opacity=0.75))
    save_plotly(fig_px, "pca_clusters_interactive.html")
    print("  ✔ PCA 2D (PNG + HTML)")

    # ── Distributions par cluster ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(axes,
                               ["Age", "Score", "Income"],
                               ["Âge", "Score", "Revenu"]):
        for c in range(k):
            df_clean[df_clean["Cluster"] == c][col].plot.kde(
                ax=ax, label=f"Cluster {c}", color=palette[c], lw=2)
        ax.set_title(f"Distribution — {title}", fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); save_fig(fig, "distributions_par_cluster.png")

    # Interactif : box plots Plotly
    fig_box = make_subplots(rows=1, cols=3,
                            subplot_titles=["Âge", "Score", "Revenu"])
    for i, col in enumerate(["Age", "Score", "Income"], start=1):
        for c in range(k):
            sub = df_clean[df_clean["Cluster"] == c][col]
            fig_box.add_trace(go.Box(y=sub, name=f"C{c}",
                                     marker_color=px_pal[c],
                                     showlegend=(i == 1)), row=1, col=i)
    fig_box.update_layout(title="Distributions par cluster",
                          template="plotly_white", height=450)
    save_plotly(fig_box, "distributions_interactive.html")
    print("  ✔ Distributions (PNG + HTML)")

    # ── Heatmap profils ──────────────────────────────────────
    profile_heat = df_clean.groupby("Cluster")[["Age", "Income", "Score"]].mean()
    profile_norm = (profile_heat - profile_heat.min()) / \
                   (profile_heat.max() - profile_heat.min())

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(profile_norm, annot=profile_heat.round(0).astype(int),
                fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Valeur normalisée"})
    ax.set_title("Profil moyen des clusters", fontsize=13, fontweight="bold")
    plt.tight_layout(); save_fig(fig, "heatmap_profils.png")

    fig_h = px.imshow(profile_norm.T,
                      text_auto=False,
                      color_continuous_scale="YlOrRd",
                      title="Heatmap — Profil moyen des clusters",
                      template="plotly_white",
                      labels={"x": "Cluster", "y": "Variable"})
    save_plotly(fig_h, "heatmap_interactive.html")
    print("  ✔ Heatmap (PNG + HTML)")

    # ── Scatter Income vs Score (Plotly 3D) ──────────────────
    fig_3d = px.scatter_3d(df_clean,
                           x="Age", y="Income", z="Score",
                           color=df_clean["Cluster"].astype(str),
                           symbol="Gender",
                           opacity=0.75,
                           title="Segmentation 3D : Âge / Revenu / Score",
                           color_discrete_sequence=px_pal,
                           labels={"color": "Cluster"},
                           template="plotly_white")
    save_plotly(fig_3d, "scatter_3d_interactive.html")
    print("  ✔ Scatter 3D interactif (HTML)")

    # ── Gender par cluster ───────────────────────────────────
    gender_pct = (df_clean.groupby(["Cluster", "Gender"])
                  .size().unstack(fill_value=0)
                  .div(df_clean.groupby("Cluster").size(), axis=0) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    gender_pct.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"],
                    edgecolor="white")
    ax.set_title("Répartition Homme/Femme par Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("%")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Genre"); ax.grid(alpha=0.2)
    plt.tight_layout(); save_fig(fig, "gender_par_cluster.png")
    print("  ✔ Répartition Genre (PNG)")


# ══════════════════════════════════════════════════════════
# ÉTAPE 7 — EXPORT EXCEL (openpyxl)
# ══════════════════════════════════════════════════════════
def export_excel(df_clean: pd.DataFrame, profile: pd.DataFrame):
    """Génère un rapport Excel multi-feuilles avec mise en forme."""
    print("\n" + "="*60)
    print("ÉTAPE 7 : Export Excel (openpyxl)")
    print("="*60)

    wb = openpyxl.Workbook()

    # ── Styles ──────────────────────────────────────────────
    HEADER_FILL  = PatternFill("solid", fgColor="2C3E50")
    HEADER_FONT  = Font(color="FFFFFF", bold=True, size=11)
    CENTER       = Alignment(horizontal="center", vertical="center")
    CLUSTER_COLORS = ["FADBD8", "D5F5E3", "D6EAF8", "FDEBD0",
                      "E8DAEF", "D5D8DC", "FDFEFE", "F9EBEA"]

    def style_header_row(ws, row=1):
        for cell in ws[row]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = CENTER

    def auto_width(ws):
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    # ── Feuille 1 : Données segmentées ──────────────────────
    ws1 = wb.active
    ws1.title = "Données Segmentées"

    display_cols = ["ID", "Age", "Gender", "Income", "Score", "Cluster"]
    df_display = df_clean[display_cols]

    for r in dataframe_to_rows(df_display, index=False, header=True):
        ws1.append(r)
    style_header_row(ws1)

    # Colorier chaque ligne selon le cluster
    for row in ws1.iter_rows(min_row=2, max_row=ws1.max_row):
        cluster_val = row[-1].value
        if isinstance(cluster_val, int) and cluster_val < len(CLUSTER_COLORS):
            fill = PatternFill("solid", fgColor=CLUSTER_COLORS[cluster_val])
            for cell in row:
                cell.fill = fill
                cell.alignment = CENTER

    auto_width(ws1)

    # ── Feuille 2 : Profil des clusters ─────────────────────
    ws2 = wb.create_sheet("Profil des Clusters")
    profile_reset = profile.reset_index()
    for r in dataframe_to_rows(profile_reset, index=False, header=True):
        ws2.append(r)
    style_header_row(ws2)
    for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row):
        for cell in row:
            cell.alignment = CENTER
    auto_width(ws2)

    # ── Feuille 3 : Statistiques globales ───────────────────
    ws3 = wb.create_sheet("Statistiques Globales")
    stats_df = df_clean[["Age", "Income", "Score"]].describe().round(2)
    for r in dataframe_to_rows(stats_df.reset_index(), index=False, header=True):
        ws3.append(r)
    style_header_row(ws3)
    auto_width(ws3)

    # ── Feuille 4 : Répartition Genre ───────────────────────
    ws4 = wb.create_sheet("Répartition Genre")
    gender_df = (df_clean.groupby(["Cluster", "Gender"])
                 .size().unstack(fill_value=0).reset_index())
    for r in dataframe_to_rows(gender_df, index=False, header=True):
        ws4.append(r)
    style_header_row(ws4)
    auto_width(ws4)

    wb.save(EXCEL_PATH)
    print(f"  ✔ Rapport Excel sauvegardé → {EXCEL_PATH}")
    print(f"    Feuilles : Données Segmentées | Profil | Statistiques | Genre")


# ══════════════════════════════════════════════════════════
# ÉTAPE 8 — SAUVEGARDE CSV
# ══════════════════════════════════════════════════════════
def save_results(df_clean: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "="*60)
    print("ÉTAPE 8 : Sauvegarde CSV")
    print("="*60)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"  ✔ Données segmentées   → {PROCESSED_PATH}")

    profile_path = os.path.join(RESULTS_DIR, "cluster_profiles.csv")
    profile.to_csv(profile_path)
    print(f"  ✔ Profil des clusters  → {profile_path}")


# ──────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_plotly(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.write_html(path, auto_open=OPEN_PLOTLY)


# ══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════
def main():
    print("\n" + "★"*60)
    print("   SOCIAL SEGMENTATION CLUSTERING — Pipeline Complet")
    print("★"*60)

    df                          = load_data(RAW_PATH)
    df_clean, X_scaled, _, _   = preprocess(df)
    best_k                      = find_optimal_k(X_scaled)
    df_clean                    = apply_kmeans(df_clean, X_scaled, best_k)
    profile                     = analyze_clusters(df_clean)
    visualize_clusters(df_clean, X_scaled)
    export_excel(df_clean, profile)
    save_results(df_clean, profile)

    print("\n" + "★"*60)
    print("  ✅  Pipeline terminé avec succès !")
    print(f"\n  📂  Résultats dans : {RESULTS_DIR}/")
    print("       ├── elbow_silhouette.png")
    print("       ├── elbow_silhouette_interactive.html  ← Plotly")
    print("       ├── pca_clusters.png")
    print("       ├── pca_clusters_interactive.html      ← Plotly")
    print("       ├── distributions_par_cluster.png")
    print("       ├── distributions_interactive.html     ← Plotly")
    print("       ├── heatmap_profils.png")
    print("       ├── heatmap_interactive.html           ← Plotly")
    print("       ├── scatter_3d_interactive.html        ← Plotly 3D")
    print("       ├── gender_par_cluster.png")
    print("       ├── cluster_profiles.csv")
    print("       └── rapport_segmentation.xlsx          ← Excel")
    print(f"\n  📄  Données traitées : {PROCESSED_PATH}")
    print("★"*60 + "\n")


if __name__ == "__main__":
    main()