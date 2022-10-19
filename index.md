# Modélisation et étude de la propagation d'une épidémie sur un graphe et application à la recherche de stratégie optimale.

Ce sujet s'inscrit dans le thème santé-prévention a été fait conjointement avec Adrien LEDOUX et Pablo MADAULE.

Pour notre TIPE, nous avons décidé de nous intéresser aux stratégies permettant de limiter ou, si possible, arêter la propagation d'un virus. Pour cela nous avons exploré différentes modélisations. Celle étudiée ici est la plus simplifiée et s'intéresse à un damier.

Pour voir la modélisation plus réaliste sur un graphe cliquez sur ce [lien](https://github.com/T-jester/TIPE-Graphe)


## Présentation de la modélisation :

- On considère une grille n\*n où chaque case représente un individu pouvant être infecté, vacciné ou susceptible.
- Les intéractions entre les cases se font suivant le [voisinage de Von Neumann](https://fr.wikipedia.org/wiki/Voisinage_de_von_Neumann).
- À chaque tour le virus se propage des infectés vers des individus susceptibles avoisinants avec une probabilité p fixée et uniforme.
- À chaque tour le joueur peut vacciner K individus susceptibles, on supposera un caractère parfait de la vaccination (une fois vacciné on ne peut plus jamais être infecté)



Voici un exemple de partie type : 
<p align="right">
<img src="https://github.com/T-jester/TIPE-Z2/blob/main/docs/assets/Partie_Type.png " width="1400px "/>
</p>

## Première stratégie

L'idée de cette première stratégie est d'encercler le cluster le plus rapidement possible. 

Nous avons réussi avec mon collègue Adrien à prouver que l'[enveloppe convexe d'un connexe est minimale pour le périmètre](https://github.com/T-jester/TIPE-Z2/blob/main/Th%C3%A9or%C3%A8me%20Ledoux.pdf). Cela combiné avec le fait que le cluster se propage presque sûrement de la forme d'une boule amène cette stratégie à être considérée comme quasi-optimale.

Comme le laisse entendre l'énoncé, nous avons eu besoin d'introduire une enveloppe convexe sur Z^2, pour cela il fallait d'abord amener un segment. Pour cela il nous avons utilisé la définition du [segment de Bresenham](https://fr.wikipedia.org/wiki/Algorithme_de_trac%C3%A9_de_segment_de_Bresenham).



![Segment de Bresenham](https://github.com/T-jester/TIPE-Z2/blob/main/docs/assets/my_segment.jpg)

Finalement grâce à l'[algorithme de Graham](https://fr.wikipedia.org/wiki/Parcours_de_Graham) qui permet de trouver les points extrémaux d'un ensemble de point (qui forment de manière unique l'ensveloppe convexe grâce au théorème de [Krein Milman](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Krein-Milman) (qui s'adapte aussi sur Z^2).


On va donc tenter de construire le lacet formant l’enveloppe convexe des infectés. Pour éviter des soucis de porosité, on choisi de construire un contour connexe. On se retrouve donc à devoir choisir entre 2 points et pour choisir on considère les différentes politiques suivantes :

## Différentes politiques

- La stratégie **Barycentre** consiste à vacciner le point le plus proche du barycentre des infectés, c'est à dire le point l'individu le "plus à risque".
- La stratégie **Monte Carlo** consiste à estimer l'espérance du nombre d'inféctés en fonction des deux choix possibles et choisir celui avec la plus faible.
- On peut aussi combiner les deux. La deuxième est meilleure mais avec une plus grande compléxité donc on peut la faire intervenir "vers la fin" de la partie.




![Exemple d'application de la stratégie Barycentre](https://github.com/T-jester/TIPE-Z2/blob/main/docs/assets/Bary_Prez.jpg)


## Résultats

Pourcentage de personnes sauvées pour p=0.5, un infecté initial qui se propage seul pendant 3 tours et 2 vaccins par tour :

- La stratégie **Barycentre** :
  - Pour une grille 30\*30 : 90%
  - Pour une grille 10\*10 : 81% 
- La stratégie **Monte Carlo** :
  - Pour une grille 10\*10 : 85%
- La stratégie vaccinant un susceptible **aléatoire** :
  - Pour une grille 30\*30 : 10%

## Autre méthode 

J'ai aussi voulu tenter une approche avec du deep-learning. Cependant une approche classique n'aboutissait pas. Je me suis donc inspiré d'un [algorithme proposé par Volodymyr Mnih et al.](https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning) ayant gagné contre de nombreux jeux d'Atari.

Même après adaptétion au problème posé, les résultats ne se sont pas encore montrés. L'une des raisons étant que je l'ai entraîné pendant "seulement" 12h ce qui est bien insignifiant comparé aux 38 jours conseillés par le papier. Aux dernières nouvelles l'algorithme parvenait à sauver 20% de la population d'une grille 30\*30 (ce qui reste 2 fois plus que la stratégie aléatoire).


## Conclusion

Sur cette modélisation (assez peu réaliste) nous avons pu trouver une stratégie quasi-optimale et très efficace. Cependant le faible réalisme reste un gros soucis. Nous avons donc proposé [une autre modélisation](https://github.com/T-jester/TIPE-Graphe) plus proche sur un graphe connexe.





Nhésitez pas à aller voir mes autre projets sur [mon GitHub](https://github.com/T-jester) :)





