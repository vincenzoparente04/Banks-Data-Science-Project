"""
APPLICATION WEB INTERACTIVE - ANALYSE BANQUES COOPÉRATIVES
Streamlit App pour explorer les résultats de l'analyse
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Analyse Banques Coopératives",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv')
    if 'Unnamed: 10' in df.columns:
        df = df.drop(columns=['Unnamed: 10'])
    
    num_cols = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df['periode'] = df['year'].apply(lambda x: 'Pré-crise' if x <= 2010 else 'Post-crise')
    return df

@st.cache_data
def load_results():
    tests = pd.read_csv('03_tests_statistiques_complets.csv')
    impacts = pd.read_csv('05_impacts_par_pays.csv')
    convergence = pd.read_csv('06_convergence_analyse.csv')
    return tests, impacts, convergence

df = load_data()
df_clean = df[['institution_name', 'year', 'country_code', 'periode', 
               'ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']].dropna()

tests_df, impacts_df, convergence_df = load_results()

# ============================================================================
# BARRE LATÉRALE - NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page:",
    ["🏠 Accueil", "📊 Tableau de bord", "🔬 Analyse Statistique", 
     "🎯 Clustering", "🌍 Analyse par Pays", "📋 Données Brutes"]
)

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    st.title("🏦 Analyse des Banques Coopératives Européennes")
    st.markdown("*Impact de la crise financière 2008 sur le business model (2005-2015)*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Observations", f"{len(df_clean):,}")
    with col2:
        st.metric("🏪 Banques uniques", df['institution_name'].nunique())
    with col3:
        st.metric("🌍 Pays couverts", df['country_code'].nunique())
    
    st.markdown("---")
    
    st.markdown("## ❓ Problématique Centrale")
    st.markdown("""
    **Comment les banques coopératives européennes ont-elles modifié leur modèle d'affaires suite à la 
    crise financière de 2008 ?**
    
    Quels changements structurels dans la composition de leurs bilans témoignent d'une réorientation 
    stratégique entre la période **pré-crise (2005-2010)** et **post-crise (2011-2015)** ?
    """)
    
    st.markdown("## 🔍 Sous-questions Clés")
    questions = {
        "1. Différences pré/post-crise ?": "✅ OUI - Toutes les variables sont significatives (p < 0.05)",
        "2. Éléments du bilan les plus changés ?": "⚠️ Actifs totaux (-73.6%), Trading (-75.9%)",
        "3. Profils de banques identifiés ?": "4 clusters avec stratégies différenciées",
        "4. Pays les plus affectés ?": "🇩🇪 Allemagne, 🇮🇹 Italie, 🇦🇹 Autriche",
        "5. Convergence vers un modèle ?": "❌ Non - Divergence observée (divergence croissante)",
        "6. Banques plus prudentes ?": "✅ OUI - Ratio RWA baisse (-2.24%)"
    }
    
    for q, a in questions.items():
        with st.expander(q):
            st.markdown(f"**{a}**")
    
    st.markdown("---")
    
    st.markdown("## 📊 Méthodologie")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Méthode 1: Tests Statistiques")
        st.markdown("""
        - Tests t de Student (Student's t-test)
        - Mesure d'effets (Cohen's d)
        - Détermination de la significativité (p-value)
        - **Résultat**: Tous les changements sont significatifs
        """)
    
    with col2:
        st.subheader("✅ Méthode 2: Clustering K-means")
        st.markdown("""
        - Normalisation StandardScaler
        - K-means clustering (k=4)
        - Caractérisation des profils
        - Analyse PCA pour visualisation
        - **Résultat**: 4 profils distincts identifiés
        """)

# ============================================================================
# PAGE 2: TABLEAU DE BORD
# ============================================================================

elif page == "📊 Tableau de bord":
    st.title("📊 Tableau de Bord Descriptif")
    
    col1, col2 = st.columns(2)
    
    with col1:
        periode_filter = st.multiselect(
            "Filtrer par période:",
            ["Pré-crise", "Post-crise"],
            default=["Pré-crise", "Post-crise"]
        )
    
    with col2:
        # Obtenir les 10 pays les plus représentés
        top_pays = df['country_code'].value_counts().head(10).index.tolist()
        pays_filter = st.multiselect(
            "Filtrer par pays (top 10):",
            top_pays,
            default=top_pays[:3]  # Seulement les 3 premiers
        )
    
    # Filtrer les données
    df_filtered = df_clean[
        (df_clean['periode'].isin(periode_filter)) & 
        (df_clean['country_code'].isin(pays_filter))
    ]
    
    st.markdown(f"**Observations affichées:** {len(df_filtered):,}")
    
    # Graphique: distribution des variables clés
    st.subheader("📈 Distribution des Variables Clés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_filtered.boxplot(column='ass_total', by='periode', ax=ax)
        ax.set_title('Actifs Totaux (Millions €)')
        ax.set_xlabel('Période')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_filtered.boxplot(column='in_roa', by='periode', ax=ax)
        ax.set_title('Rentabilité (ROA)')
        ax.set_xlabel('Période')
        st.pyplot(fig)
    
    # Statistiques descriptives
    st.subheader("📊 Statistiques Descriptives par Période")
    
    for periode in periode_filter:
        with st.expander(f"📋 {periode}"):
            stats = df_filtered[df_filtered['periode'] == periode][
                ['ass_total', 'in_roa', 'rt_rwa', 'in_roe']
            ].describe()
            st.dataframe(stats, use_container_width=True)

# ============================================================================
# PAGE 3: ANALYSE STATISTIQUE
# ============================================================================

elif page == "🔬 Analyse Statistique":
    st.title("🔬 Résultats des Tests Statistiques")
    st.markdown("Comparaison Pré-crise vs Post-crise (t-test de Student)")
    
    st.subheader("📋 Tableau Récapitulatif des Tests")
    
    # Afficher le tableau
    display_cols = ['Variable', 'Moyenne Pré-crise', 'Moyenne Post-crise', 
                   'Différence (%)', 'p-value', "Cohen's d", 'Significatif (p<0.05)']
    st.dataframe(tests_df[display_cols], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Détail pour chaque variable
    st.subheader("🔍 Analyse Détaillée par Variable")
    
    selected_var = st.selectbox(
        "Sélectionnez une variable:",
        tests_df['Variable'].tolist()
    )
    
    var_data = tests_df[tests_df['Variable'] == selected_var].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Moyenne Pré-crise", f"{var_data['Moyenne Pré-crise']:.4f}")
    with col2:
        st.metric("Moyenne Post-crise", f"{var_data['Moyenne Post-crise']:.4f}")
    with col3:
        st.metric("Variation %", f"{var_data['Différence (%)']:.2f}%")
    with col4:
        st.metric("p-value", f"{var_data['p-value']:.6f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cohen's d (Taille d'effet)", f"{var_data['Cohen\'s d']:.4f}")
    with col2:
        st.metric("Significatif ?", var_data['Significatif (p<0.05)'])
    
    # Visualisation
    st.subheader("📊 Distribution Graphique")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pre_data = df_clean[df_clean['periode'] == 'Pré-crise'][selected_var].dropna()
    post_data = df_clean[df_clean['periode'] == 'Post-crise'][selected_var].dropna()
    
    ax.hist(pre_data, alpha=0.5, label='Pré-crise', bins=30)
    ax.hist(post_data, alpha=0.5, label='Post-crise', bins=30)
    ax.set_xlabel(selected_var)
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution de {selected_var}')
    ax.legend()
    st.pyplot(fig)

# ============================================================================
# PAGE 4: CLUSTERING
# ============================================================================

elif page == "🎯 Clustering":
    st.title("🎯 Analyse de Clustering K-means")
    st.markdown("Identification de 4 profils de banques distincts")
    
    # Charger les clusters
    available_vars = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
    X = df_clean[available_vars].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean_cluster = df_clean[available_vars].notna().all(axis=1)
    df_clean.loc[df_clean_cluster, 'cluster'] = clusters
    
    # Profils des clusters
    st.subheader("👥 Profils des Clusters")
    
    cluster_profiles = df_clean.groupby('cluster')[available_vars].mean()
    st.dataframe(cluster_profiles.round(4), use_container_width=True)
    
    st.markdown("---")
    
    # Distribution par période
    st.subheader("📊 Distribution des Clusters par Période")
    
    cluster_dist = pd.crosstab(df_clean['cluster'], df_clean['periode'], margins=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Nombres absolus:")
        st.dataframe(cluster_dist, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_pct = pd.crosstab(df_clean['periode'], df_clean['cluster'], normalize='index') * 100
        cluster_pct.plot(kind='bar', ax=ax)
        ax.set_title('Distribution des clusters par période (%)')
        ax.set_ylabel('Pourcentage (%)')
        ax.legend(title='Cluster')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# PAGE 5: ANALYSE PAR PAYS
# ============================================================================

elif page == "🌍 Analyse par Pays":
    st.title("🌍 Impact par Pays")
    st.markdown("Quel pays a été le plus affecté par la crise ?")
    
    # Afficher le tableau des impacts
    st.subheader("📊 Variations par Pays")
    
    display_impacts = impacts_df[['Pays', 'Actifs Pré-crise (millions)', 
                                   'Actifs Post-crise (millions)', 'Variation (%)', 'Nb banques']].copy()
    display_impacts = display_impacts.sort_values('Variation (%)')
    
    st.dataframe(display_impacts, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Graphique interactif
    st.subheader("📈 Impact Visuel")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in display_impacts['Variation (%)']]
    ax.barh(display_impacts['Pays'], display_impacts['Variation (%)'], color=colors, alpha=0.7)
    ax.set_xlabel('Variation des actifs (%)', fontsize=12)
    ax.set_title('Impact de la crise 2008 par pays\nVariation des actifs totaux pré/post-crise', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Pays affectés
    st.subheader("⚠️ Résumé")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Top 5 Pays PLUS Affectés:**")
        top_impact = display_impacts.head(5)[['Pays', 'Variation (%)']].to_string(index=False)
        st.code(top_impact, language="text")
    
    with col2:
        st.markdown("**🟢 Top 5 Pays MOINS Affectés:**")
        top_growth = display_impacts.tail(5)[['Pays', 'Variation (%)']].to_string(index=False)
        st.code(top_growth, language="text")

# ============================================================================
# PAGE 6: DONNÉES BRUTES
# ============================================================================

elif page == "📋 Données Brutes":
    st.title("📋 Données Brutes")
    
    st.subheader("🔍 Exploration des Données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_filter = st.slider("Année:", 2005, 2015, (2005, 2015))
    with col2:
        periode_filter = st.multiselect("Période:", ["Pré-crise", "Post-crise"], 
                                       default=["Pré-crise", "Post-crise"])
    with col3:
        pays_filter = st.multiselect("Pays:", sorted(df['country_code'].unique()),
                                    default=list(sorted(df['country_code'].unique())[:5]))
    
    # Filtrer
    df_display = df[
        (df['year'] >= year_filter[0]) & (df['year'] <= year_filter[1]) &
        (df['periode'].isin(periode_filter)) &
        (df['country_code'].isin(pays_filter))
    ]
    
    st.markdown(f"**{len(df_display):,} lignes** affichées")
    
    st.dataframe(df_display, use_container_width=True, height=600)
    
    # Télécharger
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name="banques_coopératives_filtré.csv",
        mime="text/csv"
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    📊 Analyse des Banques Coopératives Européennes (2005-2015)<br>
    Données: 9,550 observations | Banques: 1,696 | Pays: 22<br>
    <b>Méthodes:</b> Tests t-Student + K-means Clustering
</div>
""", unsafe_allow_html=True)
