Projet : Reproduction RND et variante RND+Decay

Contenu :
- random-network-distillation/ : code complet utilisé
- rnd_classic_coubes_csv_videos/ : résultats RND classique avec courbes, métriques et vidéos
- rnd_decay_coubes_csv_videos/ : résultats RND+Decay avec courbes, métriques et vidéos 

Résultats principaux :
RND Classic 15M :
- best_ret = 1400
- eprew ≈ 1360
- n_rooms = 11

RND+Decay 15M :
- best_ret = 2600
- eprew ≈ 2400
- n_rooms = 6
- rnd_decay final = 0.1
- int_coeff_effective final = 0.1

Interprétation courte :
RND Classic explore davantage de salles.
RND+Decay obtient un meilleur score extrinsèque mais explore moins, ce qui suggère un passage plus rapide de l'exploration vers l'exploitation.
