# Reproduction de Random Network Distillation (RND) et proposition de la variante RND + Decay

## Présentation

Ce dépôt contient la reproduction complète de l'article :

> **Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2018).**
>
> **Exploration by Random Network Distillation**
>
> *International Conference on Learning Representations (ICLR).*

Ce projet a été réalisé dans le cadre d'un projet de Master consacré à l'apprentissage par renforcement.

En plus de reproduire fidèlement l'algorithme original **Random Network Distillation (RND)** proposé par OpenAI, ce travail introduit une variante appelée **RND + Decay**, dont l'objectif est de diminuer progressivement l'influence de la récompense intrinsèque afin d'étudier son impact sur le compromis entre exploration et exploitation.

Toutes les expériences ont été réalisées sur le jeu Atari **Montezuma's Revenge**, l'un des environnements de référence les plus difficiles en apprentissage par renforcement en raison de la rareté des récompenses.

---

# Objectifs

Les principaux objectifs de ce projet sont les suivants :

- reproduire l'implémentation originale de RND proposée par OpenAI ;
- reconstruire entièrement l'environnement expérimental utilisé dans l'article ;
- analyser le comportement exploratoire de RND ;
- proposer et implémenter la variante **RND + Decay** ;
- comparer les performances de PPO, RND et RND + Decay ;
- réaliser une analyse statistique à une seed puis à cinq seeds indépendantes.

---

# Configuration expérimentale

| Élément | Valeur |
|----------|----------|
| Environnement | MontezumaRevengeNoFrameskip-v4 |
| Framework | TensorFlow 1.15 |
| GPU | NVIDIA A100 PCIe 80 Go |
| Plateforme | RunPod |
| Nombre d'environnements parallèles | 32 |
| Budget d'entraînement | 15 millions de timesteps |
| Nombre de seeds | 5 par méthode |
| Méthodes étudiées | PPO, RND et RND + Decay |

---

# Structure du dépôt

```text
random-network-distillation+Results/

├── random-network-distillation/
│   ├── Code source original OpenAI
│   ├── Implémentation de RND + Decay
│   ├── Scripts d'entraînement
│   ├── Scripts d'analyse
│   └── Utilitaires
│
├── rnd_classic_15M/
│   ├── Checkpoints
│   ├── Journaux d'entraînement
│   ├── Vidéos
│   └── Fichiers TensorBoard
│
├── rnd_decay_15M/
│   ├── Checkpoints
│   ├── Journaux d'entraînement
│   ├── Vidéos
│   └── Fichiers TensorBoard
│
├── rnd_multiseed/
│   ├── ppo_baseline/
│   ├── rnd_classic/
│   └── rnd_decay/
│
├── multiseed_analysis_3methods/
│   ├── Courbes comparatives
│   ├── Fichiers CSV
│   ├── Tableaux récapitulatifs
│   └── Analyses statistiques
│
├── rnd_classic_courbes_csv_videos/
│
├── rnd_decay_courbes_csv_videos/
│
├── README.md
└── README_RESULTATS.txt
```

---

### Formule utilisée

```text
A_total = int_coeff × rnd_decay × A_int + ext_coeff × A_ext
```

où :

- **A_int** représente l'avantage intrinsèque ;
- **A_ext** représente l'avantage extrinsèque ;
- **rnd_decay** est un coefficient décroissant progressivement au cours de l'entraînement ;
- **int_coeff** est le coefficient de pondération de la récompense intrinsèque.

Cette modification permet :

- de favoriser une forte exploration au début de l'apprentissage ;
- de diminuer progressivement l'influence de la curiosité ;
- d'encourager davantage l'exploitation lorsque l'agent découvre des stratégies efficaces.

# Principaux résultats expérimentaux

## Expériences à une seed

| Méthode | Meilleur score | Nombre maximal de salles |
|----------|---------------|--------------------------|
| RND | 1400 | 11 |
| RND + Decay | 2600 | 6 |

La variante proposée permet d'obtenir un score maximal plus élevé, mais au prix d'une exploration plus limitée de l'environnement.

---

## Validation statistique à cinq seeds

Afin d'évaluer la robustesse des résultats, cinq graines aléatoires indépendantes ont été exécutées pour chacune des trois méthodes :

- PPO Baseline ;
- RND classique ;
- RND + Decay.

Le dépôt contient notamment :

- les courbes moyennes d'apprentissage ;
- les écarts-types ;
- les intervalles de confiance à 95 % ;
- les coefficients de variation ;
- les statistiques détaillées par seed ;
- les courbes comparatives des trois méthodes.

Cette campagne expérimentale complète les expériences à une seed en permettant d'évaluer la variabilité des performances.

---

# Principales observations

Les expériences réalisées mettent en évidence plusieurs tendances :

- le RND classique explore un plus grand nombre de salles ;
- RND + Decay exploite plus efficacement les comportements déjà appris ;
- PPO présente la plus forte variabilité entre les différentes seeds ;
- les différences observées entre les trois méthodes restent modestes compte tenu du budget expérimental disponible.

Les expériences multi-seeds doivent donc être interprétées comme une validation de robustesse et non comme une démonstration définitive de la supériorité d'une méthode.

---

# Budget expérimental

Toutes les expériences ont été réalisées sur une instance **NVIDIA A100 PCIe de 80 Go** louée sur la plateforme **RunPod**.

Le projet comprend notamment :

- les essais préliminaires ;
- la reconstruction complète de l'environnement logiciel ;
- la reproduction du pipeline original ;
- les expériences à une seed (PPO, RND et RND + Decay) ;
- la campagne de validation statistique à cinq seeds.

L'ensemble de ces expériences représente un coût de calcul estimé à environ :

- **70 USD** ;
- soit environ **50 000 FCFA**.

Une campagne expérimentale de plus grande ampleur (plusieurs dizaines de seeds ou plusieurs centaines de millions de timesteps) aurait nécessité un budget de calcul nettement supérieur.

---

# Reproductibilité

Ce dépôt contient l'ensemble des éléments nécessaires à la reproduction des expériences :

- le code source original ;
- les modifications apportées à l'algorithme ;
- les scripts d'entraînement ;
- les scripts d'analyse ;
- les journaux d'entraînement ;
- les checkpoints ;
- les fichiers TensorBoard ;
- les métriques au format CSV ;
- les analyses statistiques ;
- les courbes d'apprentissage ;
- les vidéos de gameplay.

L'ensemble des résultats présentés dans le rapport peut ainsi être reproduit et vérifié.

---

# Auteur

**Yarame BA**

Master Science des Données et Applications  
Option : Ingénierie des Données et Intelligence Artificielle (IIA)

Université Iba Der Thiam de Thiès (UIDT)

Année universitaire 2025–2026

---

# Référence bibliographique

Si vous utilisez ce dépôt, merci de citer l'article original :

```bibtex
@inproceedings{burda2018exploration,
  title={Exploration by Random Network Distillation},
  author={Burda, Yuri and Edwards, Harrison and Pathak, Deepak and Storkey, Amos and Darrell, Trevor and Efros, Alexei A.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```