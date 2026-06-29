============================================================
REPRODUCTION DE RANDOM NETWORK DISTILLATION (RND)
ET VARIANTE RND + DECAY
============================================================

Auteur : Yarame BA

Université Iba Der Thiam de Thiès (UIDT)

Master Science des Données et Applications
Option : Ingénierie des Données et Intelligence Artificielle (IIA)

============================================================
CONTENU DU DOSSIER
============================================================

Ce dossier regroupe l'ensemble des fichiers produits lors du projet de
reproduction de l'article :

Burda Y. et al.
"Exploration by Random Network Distillation"
(ICLR, 2019)

Le dépôt contient notamment :

• random-network-distillation/
    - code source original OpenAI
    - implémentation de la variante RND + Decay
    - scripts d'entraînement
    - scripts d'analyse

• rnd_classic_15M/
    - expérience complète RND classique (15 millions de timesteps)

• rnd_decay_15M/
    - expérience complète RND + Decay (15 millions de timesteps)

• rnd_classic_courbes_csv_videos/
    - courbes TensorBoard
    - métriques CSV
    - vidéos de gameplay

• rnd_decay_courbes_csv_videos/
    - courbes TensorBoard
    - métriques CSV
    - vidéos de gameplay

• rnd_multiseed/
    - campagnes expérimentales à cinq seeds
    - PPO baseline
    - RND classique
    - RND + Decay

• multiseed_analysis_3methods/
    - courbes comparatives
    - statistiques
    - tableaux récapitulatifs
    - intervalles de confiance
    - analyses multi-seeds

============================================================
RÉSULTATS DES EXPÉRIENCES À UNE SEED
============================================================

RND classique (15 millions de timesteps)

• Meilleur score obtenu (best_ret) : 1400
• Score moyen des meilleurs épisodes (eprew) : ≈ 1360
• Nombre maximal de salles explorées : 11

------------------------------------------------------------

RND + Decay (15 millions de timesteps)

• Meilleur score obtenu (best_ret) : 2600
• Score moyen des meilleurs épisodes (eprew) : ≈ 2400
• Nombre maximal de salles explorées : 6
• Coefficient final de décroissance : 0.10
• Coefficient intrinsèque effectif final : 0.10

============================================================
VALIDATION STATISTIQUE
============================================================

Une seconde campagne expérimentale a été réalisée afin d'évaluer la
robustesse des résultats.

Trois méthodes ont été comparées :

• PPO Baseline
• RND classique
• RND + Decay

Pour chacune d'elles, cinq graines aléatoires indépendantes ont été
exécutées.

Les analyses disponibles comprennent :

• courbes moyennes ;
• écarts-types ;
• intervalles de confiance à 95 % ;
• coefficients de variation ;
• statistiques par seed ;
• comparaisons entre les trois méthodes.

============================================================
INTERPRÉTATION GÉNÉRALE
============================================================

Les expériences montrent que :

• le RND classique explore un plus grand nombre de salles ;

• RND + Decay obtient un score maximal plus élevé mais réduit
  progressivement l'exploration au profit de l'exploitation ;

• les expériences multi-seeds mettent en évidence une forte variabilité
  entre les différentes graines aléatoires, caractéristique de
  Montezuma's Revenge ;

• les différences observées entre les méthodes doivent être interprétées
  comme des tendances expérimentales dans les conditions de calcul
  retenues.

============================================================