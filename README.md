# TIPE-2022
Tirte : Modélisation et étude de la propagation d'une épidémie sur un graphe et application à la recherche de stratégie vaccinale.




 Pour tester l'algorithme :


Créez un environement avec, par exemple, la commande :

env = Environment_Z2(grid_length = 30, nb_init = 1, advantage = 3, n_infections_per_step = 1, proba = 0.5, nb_vaccin = 2)

qui considère une grille de 30*30 avec un seul infécté initial (pas encore présent sur la grille) et qui laissera à l'initialisation le virus se propager 
3 tours avant de pouvoir commencer à vacciner. On peut vacciner 2 personnes par tour puis l'infection se propage "une fois" avec une probabilité de 50%.

Pour l'affichage du jeu avec possibilité d'utiliser une IA, tapez env.jeu_aide_IA(IA_Cvx) dans la console et executez.

Les boutons, de droite à gauche, vous permettent de :
-vacciner une case suivant l'algorithme donné en entré,
-infecter pour un tour suivant la probabilité donné à la création de l'environment,
-sauvegarder l'image présente en face de vous (/!\ en sauvegarder une nouvelle écrasera la précédente, changez le nom de sauvegarde à la ligne 312 de Jeu_Z2 ou celui de l'image déjà téléchargée pour en ajouter une nouvelle),
-recommencer une nouvelle partie si celle en cours vous ennuie :) (pour quitter il suffit de fermer la page...)


### Pour le jeu solo, problème dans la fermeture du fichier... fonctionne si on attend jusqu'à la fin mais si par malheur on ferme la fenêtre avant 
### il faut redémarer la consolle ;(




Voici un exemple de test de probabilité à faire pour tester l'efficacité de l'algorithme et comparer différentes politiques de choix :

env = Environment_Z2(grid_length = 10, nb_init = 1, advantage = 3, n_infections_per_step = 1, proba = 0.5, nb_vaccin = 2)

n_eval = 10000 ; n_vaccin = 2

tot = 0
mean_n_infect_ini = 0
IA = IAs.IA_Cvx_Bary

### /!\ IAs.IA_Cvx_Combinaison prend en argument non pas la grille mais l'environnement directement, il faut donc modifier en IA(env) ligne 50

for _ in range(n_eval) :
    env.clear()
    env.start()
    mean_n_infect_ini += env.n_infected
    while not env.is_finished() :
        for _ in range(n_vaccin) :
            (x,y) = IA(grille)
            env.step(x,y)
        env.spread()
    tot+=env.n_infected



Pour le pourcentage de personnes infectés :
tot/n_eval/100

# Quelques valeurs :

Pour une ancienne version de IA_Cvx_Bary :
grid_length=30 & proba=0.5 => 13% = 117 infectés en moyenne
grid_length=8 & proba=1 => 34% = 22 infectés en moyenne

(La version plus récente s'approche plus des 10% aux derniers calculs)


IA_Cvx_combinaison : grid_length=8 & proba=0.5 =>
tot/n_eval/64 -> 0.30234375 -> 19.35 infectés en moyenne
grid_length = 10 => 0.26778 infectés

onte_Carlo_Graham : grid_length=8 & proba=0.5 =>
tot/n_eval/64 -> 0.3078125 -> 19.7 infectés en moyenne
0.27226 pour grid_length = 10

Pour le nombre d'inféctés initiaux
mean_n_infect_ini/n_eval
idem => 10 infectés initiaux

