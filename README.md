# 🏦 Analyse des Banques Coopératives Européennes

## 📋 Vue d'ensemble du Projet

**Objectif:** Analyser l'évolution du business model des banques coopératives européennes avant et après la crise financière de 2008 (2005-2015).

**Données:** 9,550 observations | 1,696 banques uniques | 22 pays européens

---

## 🎯 Questions de Recherche Clés

1. **Existe-t-il des différences significatives pré/post-crise ?**
   - ✅ **OUI** - Toutes les variables sont significatives (p < 0.05)

2. **Quels éléments du bilan ont le plus changé ?**
   - ⚠️ Actifs totaux : **-73.6%** (réduction drastique)
   - ⚠️ Actifs de trading : **-75.9%** (abandon des marchés)
   - ⚠️ Revenus de trading : **-66.5%** (moins spéculatif)
   - ⚠️ Rentabilité (ROE) : **-26.6%** (moins profitable)

3. **Peut-on identifier des groupes de banques aux stratégies différentes ?**
   - ✅ **4 clusters** identifiés avec profils distincts

4. **Quels pays/régions ont été les plus affectés ?**
   - 🇩🇪 Allemagne : **-72%**
   - 🇮🇹 Italie : **-69%**
   - 🇦🇹 Autriche : **-67%**

5. **Existe-t-il une convergence vers un modèle plus similaire ?**
   - ❌ **Non** - Divergence observée (les banques deviennent plus différentes)

6. **Les banques sont-elles devenues plus prudentes ?**
   - ✅ **OUI** - Ratio RWA baisse (-2.24%), conformité Bâle III

---

## 📁 Structure des Fichiers

```
/home/ubuntu/Bureau/testData/
├── test.py                                    # Script initial (tests + clustering)
├── analyse_complete.py                        # Version améliorée avec toutes analyses
├── app_streamlit.py                           # Application Web Interactive
│
├── DATA INPUTS:
│   ├── Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv
│   └── Theme4_coop_zoom_data.xlsx - Dictionary_of_variables.csv
│
├── OUTPUTS - Fichiers CSV:
│   ├── 03_tests_statistiques_complets.csv     # Tests t-Student (Méthode 1)
│   ├── 04_cluster_profiles.csv                # Profils clusters K-means
│   ├── 05_impacts_par_pays.csv                # Variations par pays
│   └── 06_convergence_analyse.csv             # Analyse convergence
│
├── OUTPUTS - Graphiques:
│   ├── 07_impacts_par_pays.png                # Carte impact par pays
│   ├── 08_evolution_temporelle.png            # Timeline 2005-2015
│   └── 09_clusters_par_periode.png            # Distribution clusters
│
├── myenv/                                     # Environnement Python (venv)
└── README.md                                  # Ce fichier
```

---

## 🔬 Méthodologie

### ✅ Méthode 1: Tests Statistiques Comparatifs

**Objectif:** Valider les changements significatifs entre les deux périodes

**Technique:** Tests t de Student (Student's t-test)
- Hypothèse H0 : Pas de différence significative pré/post-crise
- Hypothèse H1 : Différence significative

**Mesures:**
- **t-statistic** : Mesure la différence relative à la variabilité
- **p-value** : Probabilité que la différence soit due au hasard (< 0.05 = significatif)
- **Cohen's d** : Taille d'effet (mesure pratique de l'ampleur du changement)

**Résultats:**
```
Variable        | Pré-crise | Post-crise | Variation | p-value | Significatif
ass_total       | 20,072.6  | 5,295.2    | -73.6%    | <0.001  | ✅ OUI
ass_trade       | 7,183.5   | 1,731.4    | -75.9%    | <0.001  | ✅ OUI
inc_trade       | 25.9      | 8.7        | -66.5%    | 0.017   | ✅ OUI
in_roa          | 0.0054    | 0.0047     | -13.9%    | <0.001  | ✅ OUI
rt_rwa          | 0.612     | 0.599      | -2.2%     | 0.016   | ✅ OUI
in_roe          | 0.073     | 0.054      | -26.6%    | <0.001  | ✅ OUI
in_trade        | -0.016    | 0.035      | -312%     | <0.001  | ✅ OUI
```

### ✅ Méthode 2: Clustering K-means

**Objectif:** Identifier des groupes de banques avec profils similaires

**Technique:** K-means clustering (k=4)
1. Normalisation des données (StandardScaler)
2. Clustering itératif sur 7 variables clés
3. Caractérisation des profils moyens

**Variables utilisées:**
- `ass_total` : Actifs totaux
- `ass_trade` : Actifs de trading
- `inc_trade` : Revenus de trading
- `in_roa` : Rentabilité des actifs
- `rt_rwa` : Ratio actifs pondérés en risque
- `in_roe` : Rentabilité des fonds propres
- `in_trade` : Poids du trading

**Résultats:** 4 profils découverts avec distributions différentes pré/post-crise

---

## 🚀 Comment Utiliser

### 1. Exécuter l'analyse complète

```bash
# Aller dans le répertoire
cd /home/ubuntu/Bureau/testData

# Activer l'environnement
source myenv/bin/activate

# Lancer l'analyse
python analyse_complete.py
```

**Sortie:** Génère 7 fichiers CSV + graphiques PNG dans le répertoire courant

### 2. Lancer l'application Web Interactive

```bash
# Depuis le répertoire testData avec l'environnement activé
streamlit run app_streamlit.py
```

**URL:** Streamlit ouvrira automatiquement à `http://localhost:8501`

**Pages disponibles:**
- 🏠 **Accueil** : Présentation & problématique
- 📊 **Tableau de bord** : Visualisations filtrables (pays, période)
- 🔬 **Analyse Statistique** : Détail des tests t
- 🎯 **Clustering** : Exploration des profils
- 🌍 **Analyse par Pays** : Impact régional
- 📋 **Données Brutes** : Export CSV

### 3. Interpréter les Résultats

Pour chaque variable, regarder:
1. **p-value < 0.05** → Changement significatif ✅
2. **Cohen's d** → Taille d'effet (0.2=petit, 0.5=moyen, 0.8=grand)
3. **Variation (%)** → Ampleur du changement

---

## 📊 Dictionnaire des Variables

| Variable | Description | Signification |
|----------|-------------|---------------|
| `ass_total` | Actifs totaux (millions €) | Taille de la banque |
| `ass_trade` | Actifs de trading (millions €) | Part orientée marchés financiers |
| `inc_trade` | Revenus de trading (millions €) | Bénéfices du trading |
| `in_roa` | Retour sur actifs (%) | Efficacité d'utilisation des actifs |
| `rt_rwa` | Ratio actifs pondérés risque | Risque par unité d'actifs |
| `in_roe` | Retour sur fonds propres (%) | Rentabilité commerciale |
| `in_trade` | Poids trading / revenus totaux (%) | Dépendance au trading |

---

## 💡 Interprétation Métier

### Avant la crise (2005-2010): Modèle Agressif
- ✅ Banques grandes et diversifiées
- ✅ Exposition importante aux marchés financiers
- ✅ Revenus élevés du trading
- ✅ Levier financier élevé

### Après la crise (2011-2015): Modèle Prudent
- ✅ Réduction drastique des actifs (-73.6%)
- ✅ Retrait des marchés financiers (-75.9%)
- ✅ Réduction des revenus spéculatifs (-66.5%)
- ✅ Renforcement des fonds propres (conformité Bâle III)
- ✅ Diminution du risque (RWA -2.24%)

**Conclusion:** La crise a forcé une **réorientation stratégique majeure** vers un **modèle plus prudent et soutenable**

---

## 📝 Rapport (À Rédiger - Structure)

### Structure recommandée (15 pages max):

1. **Introduction (2 p.)**
   - Contexte crise 2008
   - Importance des banques coopératives
   - Problématique centrale

2. **Données & Technologie (1 p.)**
   - Source des données
   - 9,550 observations, 1,696 banques, 22 pays
   - Variables disponibles

3. **Méthodologie (2 p.)**
   - Tests t de Student (Méthode 1)
   - K-means Clustering (Méthode 2)
   - Normalisation StandardScaler
   - Justification des choix

4. **Analyse Descriptive (2-3 p.)**
   - Statistiques globales pré/post-crise
   - Distribution par pays
   - Évolution temporelle (graphiques)

5. **Analyse Avancée & Résultats (3-4 p.)**
   - Résultats tests statistiques (tableau)
   - Profils de banques (clusters)
   - Impact par pays
   - Analyse convergence

6. **Interprétation & Insights Métier (2 p.)**
   - Réponses aux 6 sous-questions
   - Conformité Bâle III
   - Changements stratégiques

7. **Conclusion (1 p.)**
   - Résumé changements majeurs
   - Perspectives futures

8. **Annexes**
   - Code complet (analyse_complete.py)
   - Graphiques supplémentaires
   - Dictionnaire données

---

## 🎬 Présentation (8 minutes)

### Structure de la soutenance:

1. **Intro (1 min):** Problématique + données
2. **Méthodes (1 min):** Tests t + K-means (simple)
3. **Résultats (3 min):** Répondre aux 6 questions clés
4. **Démo App (2 min):** Naviguer les 6 pages Streamlit
5. **Conclusion (1 min):** Key takeaways + limitations

**Points clés à mettre en avant:**
- ✅ 2 méthodes complémentaires (tests + clustering)
- ✅ Tous les changements sont **significatifs** (p < 0.05)
- ✅ **4 profils** de banques identifiés
- ✅ Pays **les plus affectés** (Germany, Italy, Austria)
- ✅ Banques **plus prudentes** (RWA baisse)
- ✅ **Divergence** croissante entre banques

---

## 🛠️ Commandes Utiles

```bash
# Activer l'environnement
source myenv/bin/activate

# Exécuter l'analyse
python analyse_complete.py

# Lancer l'app Streamlit
streamlit run app_streamlit.py

# Voir l'historique des fichiers générés
ls -lht *.csv *.png 2>/dev/null | head -20

# Voir les résultats des tests
cat 03_tests_statistiques_complets.csv | column -t -s,

# Vérifier les dépendances
pip list | grep -E "pandas|matplotlib|scikit-learn|streamlit"
```

---

## ✅ Checklist Avant Rendu

- [ ] **Scripts**
  - [ ] `test.py` fonctionne
  - [ ] `analyse_complete.py` génère tous les fichiers
  - [ ] `app_streamlit.py` lancée sans erreurs

- [ ] **Fichiers générés**
  - [ ] 03_tests_statistiques_complets.csv ✓
  - [ ] 04_cluster_profiles.csv ✓
  - [ ] 05_impacts_par_pays.csv ✓
  - [ ] 06_convergence_analyse.csv ✓
  - [ ] Graphiques PNG (07-09) ✓

- [ ] **Rapport (15 pages max)**
  - [ ] Problématique claire
  - [ ] 2 méthodes expliquées
  - [ ] Réponses aux 6 sous-questions
  - [ ] Lien vers l'app interactive

- [ ] **App Interactive**
  - [ ] 6 pages fonctionnelles
  - [ ] Filtres pays/période
  - [ ] Visualisations interactives
  - [ ] Export CSV possible

- [ ] **Soutenance (8 min)**
  - [ ] Slides préparées
  - [ ] Démo app prête
  - [ ] Timing respecté

---

## 📞 Support

**Problèmes courants:**

1. **"Module pandas not found"**
   ```bash
   source myenv/bin/activate
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

2. **"Streamlit not installed"**
   ```bash
   source myenv/bin/activate
   pip install streamlit
   ```

3. **"Data file not found"**
   - Vérifier que `Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv` existe
   - Vérifier le chemin courant: `pwd`

4. **"Port 8501 already in use"**
   ```bash
   streamlit run app_streamlit.py --server.port 8502
   ```

---

**Bonne chance ! 🚀**

*Dernière mise à jour: 13 janvier 2026*
