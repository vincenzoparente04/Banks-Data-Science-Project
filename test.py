"""
PROJET DATA SCIENCE - ANALYSE DES BANQUES COOPÉRATIVES EUROPÉENNES
Analyse de l'évolution du business model avant/après la crise financière 2008
Période: 2005-2015 | Pré-crise: 2005-2010 | Post-crise: 2011-2015
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PHASE 1: CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================================

print("="*80)
print("PHASE 1: CHARGEMENT DES DONNÉES")
print("="*80)

# Charger les données
df = pd.read_csv('Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv')

# Supprimer la colonne inutile si elle existe
if 'Unnamed: 10' in df.columns:
    df = df.drop(columns=['Unnamed: 10'])

# Colonnes financières à convertir
num_cols = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']

# Remplacer les virgules par des points et convertir en float
for col in num_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print(f"\n📊 Dimensions du dataset: {df.shape}")
print(f"   - Observations: {df.shape[0]:,}")
print(f"   - Variables: {df.shape[1]}")

print("\n📋 Premières lignes:")
print(df.head())

print("\n📅 Période temporelle:")
print(f"   - Années: {df['year'].min()} à {df['year'].max()}")
print(f"   - Distribution par année:")
print(df['year'].value_counts().sort_index())

print("\n🌍 Pays couverts:")
print(df['country_code'].value_counts())

print("\n🏦 Nombre de banques uniques:", df['institution_name'].nunique())

# Vérifier les valeurs manquantes
print("\n❌ Valeurs manquantes (%):")
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(missing[missing > 0].head(10))

# ============================================================================
# PHASE 2: PRÉPARATION DES DONNÉES
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: PRÉPARATION DES DONNÉES")
print("="*80)

# Créer la variable période
df['periode'] = df['year'].apply(lambda x: 'Pre-crise' if x <= 2010 else 'Post-crise')

print("\n📊 Distribution par période:")
print(df['periode'].value_counts())

# Sélectionner les variables clés pour l'analyse
key_vars = [
    'ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade'
]

# Variables disponibles dans ton dataset (à adapter si besoin)
available_vars = [col for col in key_vars if col in df.columns]

print(f"\n✅ Variables clés disponibles: {len(available_vars)}/{len(key_vars)}")
print(available_vars)

# Créer le dataset pour l'analyse (retirer les NaN)
df_clean = df[['institution_name', 'year', 'country_code', 'periode'] + available_vars].copy()
df_clean = df_clean.dropna(subset=available_vars)

print(f"\n🧹 Après nettoyage: {df_clean.shape[0]:,} observations")

# ============================================================================
# PHASE 3: ANALYSE DESCRIPTIVE
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: ANALYSE DESCRIPTIVE")
print("="*80)

# Statistiques descriptives par période
print("\n📊 STATISTIQUES DESCRIPTIVES PAR PÉRIODE\n")

for var in available_vars:
    print(f"\n{'='*60}")
    print(f"Variable: {var}")
    print(f"{'='*60}")
    
    stats_by_period = df_clean.groupby('periode')[var].describe()
    print(stats_by_period)

# Créer des visualisations comparatives
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution des variables clés: Pré-crise vs Post-crise', 
             fontsize=16, fontweight='bold')

for idx, var in enumerate(available_vars[:6]):  # 6 premières variables
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Boxplot comparatif
    df_clean.boxplot(column=var, by='periode', ax=ax)
    ax.set_title(f'{var}')
    ax.set_xlabel('Période')
    plt.sca(ax)
    plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('01_distribution_variables.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: 01_distribution_variables.png")

# ============================================================================
# PHASE 4: TESTS STATISTIQUES (Méthode 1)
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: TESTS STATISTIQUES COMPARATIFS")
print("="*80)

print("\n🔬 Test t de Student: Comparaison Pré-crise vs Post-crise\n")

results_tests = []

for var in available_vars:
    # Séparer les données
    pre_crise = df_clean[df_clean['periode'] == 'Pre-crise'][var].dropna()
    post_crise = df_clean[df_clean['periode'] == 'Post-crise'][var].dropna()
    
    # Test t de Student
    t_stat, p_value = stats.ttest_ind(pre_crise, post_crise)
    
    # Cohen's d (mesure de l'effet)
    cohens_d = (pre_crise.mean() - post_crise.mean()) / np.sqrt(
        ((len(pre_crise)-1) * pre_crise.std()**2 + (len(post_crise)-1) * post_crise.std()**2) / 
        (len(pre_crise) + len(post_crise) - 2)
    )
    
    # Interprétation
    significatif = "✅ OUI" if p_value < 0.05 else "❌ NON"
    
    results_tests.append({
        'Variable': var,
        'Moyenne Pré-crise': pre_crise.mean(),
        'Moyenne Post-crise': post_crise.mean(),
        'Différence (%)': ((post_crise.mean() - pre_crise.mean()) / pre_crise.mean() * 100),
        't-statistic': t_stat,
        'p-value': p_value,
        "Cohen's d": cohens_d,
        'Significatif (p<0.05)': significatif
    })
    
    print(f"\n{'='*60}")
    print(f"Variable: {var}")
    print(f"{'='*60}")
    print(f"Moyenne Pré-crise:  {pre_crise.mean():.6f}")
    print(f"Moyenne Post-crise: {post_crise.mean():.6f}")
    print(f"Différence (%):     {((post_crise.mean() - pre_crise.mean()) / pre_crise.mean() * 100):.2f}%")
    print(f"t-statistic:        {t_stat:.4f}")
    print(f"p-value:            {p_value:.6f}")
    print(f"Cohen's d:          {cohens_d:.4f}")
    print(f"Significatif:       {significatif}")

# Sauvegarder les résultats
df_results = pd.DataFrame(results_tests)
df_results.to_csv('02_tests_statistiques.csv', index=False)
print("\n✅ Résultats sauvegardés: 02_tests_statistiques.csv")

# ============================================================================
# PHASE 5: CLUSTERING (Méthode 2)
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: CLUSTERING K-MEANS")
print("="*80)

# Préparer les données pour le clustering
X = df_clean[available_vars].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Déterminer le nombre optimal de clusters (méthode du coude)
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualiser la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Nombre de clusters (k)', fontsize=12)
plt.ylabel('Inertie', fontsize=12)
plt.title('Méthode du coude - Détermination du nombre optimal de clusters', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('03_elbow_method.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: 03_elbow_method.png")

# Clustering avec k=4 (à ajuster selon le graphique)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_clean_for_clustering = df_clean[available_vars].dropna()
clusters = kmeans.fit_predict(X_scaled)
df_clean.loc[df_clean_for_clustering.index, 'cluster'] = clusters

print(f"\n📊 Distribution des clusters:")
print(df_clean['cluster'].value_counts().sort_index())

# Analyser les clusters par période
print("\n📊 Distribution des clusters par période:")
cluster_period = pd.crosstab(df_clean['cluster'], df_clean['periode'], normalize='columns') * 100
print(cluster_period)

# Caractériser les clusters
print("\n📊 CARACTÉRISATION DES CLUSTERS (moyennes):\n")
cluster_profiles = df_clean.groupby('cluster')[available_vars].mean()
print(cluster_profiles)

cluster_profiles.to_csv('04_cluster_profiles.csv')
print("\n✅ Profils des clusters sauvegardés: 04_cluster_profiles.csv")

# Visualisation PCA avec clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=clusters, cmap='viridis', 
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('Visualisation des clusters (PCA)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('05_clusters_pca.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: 05_clusters_pca.png")

# ============================================================================
# PHASE 6: ANALYSE PAR PAYS
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: ANALYSE PAR PAYS")
print("="*80)

# Comparaison par pays
print("\n🌍 Moyenne des variables par pays et période:\n")

for var in available_vars[:3]:  # 3 premières variables
    print(f"\n{'='*60}")
    print(f"Variable: {var}")
    print(f"{'='*60}")
    pivot = df_clean.pivot_table(values=var, 
                                  index='country_code', 
                                  columns='periode', 
                                  aggfunc='mean')
    pivot['Variation (%)'] = ((pivot['Post-crise'] - pivot['Pre-crise']) / 
                               pivot['Pre-crise'] * 100)
    print(pivot)

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("\n" + "="*80)
print("RÉSUMÉ DE L'ANALYSE")
print("="*80)

print("""
✅ FICHIERS GÉNÉRÉS:
   1. 01_distribution_variables.png - Distributions comparatives
   2. 02_tests_statistiques.csv - Résultats des tests t
   3. 03_elbow_method.png - Méthode du coude
   4. 04_cluster_profiles.csv - Profils des clusters
   5. 05_clusters_pca.png - Visualisation PCA

📊 RÉSULTATS CLÉS À INTERPRÉTER:
   - Quelles variables ont significativement changé ?
   - Combien de profils de banques identifiés ?
   - Quels pays ont été les plus affectés ?
   - Les banques sont-elles devenues plus prudentes ?

📝 PROCHAINES ÉTAPES:
   1. Interpréter les résultats
   2. Rédiger le rapport (15 pages max)
   3. Créer l'application Streamlit
   4. Préparer la soutenance (8 min)
""")

print("\n" + "="*80)
print("ANALYSE TERMINÉE !")
print("="*80)
