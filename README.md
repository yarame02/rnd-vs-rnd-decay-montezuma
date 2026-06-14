# Reproduction de Random Network Distillation (RND) et variante RND+Decay

## Présentation

Ce projet a été réalisé dans le cadre d'un projet de recherche en apprentissage par renforcement visant à reproduire les résultats de l'article :

> Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T. & Efros, A. A. (2018)
>
> **Exploration by Random Network Distillation**

L'étude porte sur le jeu Atari **Montezuma's Revenge**, un environnement réputé difficile en raison de la rareté des récompenses et de la nécessité d'une exploration efficace.

En complément de la reproduction de l'algorithme original **Random Network Distillation (RND)**, une variante appelée **RND+Decay** a été développée afin d'étudier l'impact d'une diminution progressive du poids de la récompense intrinsèque au cours de l'apprentissage.

---

## Objectifs

Les principaux objectifs de ce travail sont :

- Reproduire l'algorithme RND proposé par OpenAI ;
- Vérifier la capacité de l'algorithme à explorer efficacement Montezuma's Revenge ;
- Implémenter une variante RND+Decay ;
- Comparer les performances des deux approches ;
- Étudier le compromis entre exploration et exploitation.

---

## Configuration expérimentale

| Élément | Valeur |
|----------|----------|
| Environnement | MontezumaRevengeNoFrameskip-v4 |
| Algorithme de base | Random Network Distillation (RND) |
| Framework | TensorFlow 1.15 |
| GPU | NVIDIA A100 80 Go |
| Plateforme | RunPod |
| Nombre d'environnements parallèles | 32 |
| Budget d'entraînement | ≈ 15 millions de frames |
| Jeu étudié | Montezuma's Revenge |

---

## Structure du dépôt

```text
random-network-distillation/
│
├── Code source original et modifications apportées
│
rnd_classic_15M/
│
├── Logs d'entraînement
├── Fichiers de sauvegarde
├── Métriques expérimentales
│
rnd_decay_15M/
│
├── Logs d'entraînement
├── Fichiers de sauvegarde
├── Métriques expérimentales
│
rnd_classic_coubes_csv_videos/
│
├── Courbes d'apprentissage
├── Fichiers CSV
├── Vidéos de gameplay
│
rnd_decay_courbes_csv_videos/
│
├── Courbes d'apprentissage
├── Fichiers CSV
├── Vidéos de gameplay
│
README_RESULTATS.txt
└── Résumé détaillé des résultats
```

---

## Modifications apportées

Une variante de l'algorithme original a été développée :

### RND+Decay

L'idée consiste à réduire progressivement l'influence de la récompense intrinsèque :

```text
A = (λ × RND_Decay) × A_int + A_ext
```

où :

- `A_int` représente l'avantage intrinsèque ;
- `A_ext` représente l'avantage extrinsèque ;
- `RND_Decay` diminue progressivement au cours de l'entraînement.

Cette modification vise à :

- favoriser l'exploration au début ;
- encourager davantage l'exploitation lorsque l'agent maîtrise mieux l'environnement.

---

## Résultats principaux

### RND Classique

- Meilleure récompense observée : ≈ 1400
- Récompense moyenne des meilleurs épisodes : ≈ 1360
- Nombre maximal de salles explorées : **11**

### RND+Decay

- Meilleure récompense observée : ≈ 2600
- Récompense moyenne des meilleurs épisodes : ≈ 2400
- Nombre maximal de salles explorées : **6**
- Coefficient de décroissance final : **0.1**

---

## Comparaison des approches

| Méthode | Récompense | Exploration |
|----------|----------|----------|
| RND Classique | Plus faible | Plus importante |
| RND+Decay | Plus élevée | Plus limitée |

Observation principale :

- Le RND classique explore davantage l'environnement.
- Le RND+Decay obtient des récompenses plus élevées mais visite moins de nouvelles salles.
- Le mécanisme de décroissance favorise progressivement l'exploitation au détriment de l'exploration.

---

## Contenu du dépôt

Le dépôt contient :

- le code source complet ;
- les modifications apportées à l'algorithme ;
- les journaux d'entraînement ;
- les métriques au format CSV ;
- les courbes d'apprentissage ;
- les vidéos de gameplay ;
- les résultats expérimentaux.

L'objectif est de garantir la reproductibilité complète des expériences réalisées.

---

## Auteur

**Yarame BA**

Master Science des Données et Applications (IIA)

Université Iba Der Thiam de Thiès (UIDT)

Projet de reproduction et d'analyse de Random Network Distillation sur Montezuma's Revenge.