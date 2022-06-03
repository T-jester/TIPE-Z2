## Imports

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.widgets import Button


## Outils Pour Le Calcul De L'Enveloppe Convexe

class Calcul_Convexe(object) :
    def determinant(A,B,C):
        """déterminant de la famille de vecteurs (AB,AC) dans
        la base (x,y)
        """

        x1,y1 = A ;x2,y2 = B ;x3,y3 = C
        return((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))


    def angle(Z):
        """
        pour calculer l'angle et l'écartement à un point
        de référence (x0,y0)
        """

        (x0,y0),(x1,y1) = Z
        if x0==x1 :
            if y0==y1:
                return -10,0
            return ((np.pi/2),y0-y1)
        return (np.arctan((y1-y0)/(x1-x0)),y0-y1)



    def mint(t,n):
        """
        pour chercher le pivot
        """

        i_min = 0
        for k in range(1,n):
            t0 = t[k]
            ti = t[i_min]
            if t0<ti :
                i_min = k
        return i_min


    def echange(t,i,n):
        """
        pour placer le pivot en tête de liste et préparer la
        liste au tri
        """

        t[i],t[0] = t[0], t[i]
        for k in range(n-1,-1,-1):
            t[k] = [t[0],t[k]]


    def retour (t,n):
        """
        renormalise les listes après le tri
        """
        for k in range(n):
            t[k] = t[k][1]





    ## "Calcul" des coordonnées des sommets de l'enveloppe convexe

    def graham (t):
        """
        Renvoie les points extrémaux de t selon l'algorithme
        de Graham
        """

        n = len(t)

        if n == 0 :
            raise IndexError

        Calcul_Convexe.echange(t,Calcul_Convexe.mint(t,n),n)
        t.sort(key = Calcul_Convexe.angle) #trier t (angle croissant)
        Calcul_Convexe.retour(t,n)


        if n >=2 :
            stack = [t[0],t[1]] #stack est l'ensemble des sommets de l'enveloppe convexe : initialement un ensemble à deux éléments distinct a pour enveloppe convexe ces mêmes points

            for i in range(2,n):

                while (len(stack)>1 and Calcul_Convexe.determinant(t[i],stack[-2],stack[-1])<0):
                    stack.pop() # si t[i] est à droite alors stack[-1] est à gauche du segment [stack[-2],t[i]]
                stack += [t[i]]

        else:
            stack = [t[0]]

        if len(stack)>2 and Calcul_Convexe.determinant (stack[-1],stack[-2],t[0])==0:
            stack.pop() # si le premier et les deux derniers points de l'ensemble sont alignés

        return stack




    ## Tracer un segment suivant la methode de Bresenham :

    def bresenham (A,B) :
        """
        Renvoie les points de Z2 formant le segment de Bresenham
        entre A et B
        """

        # on différencie les 8 octants du plan
        x1,y1 = A ; x2,y2 = B
        l = []
        dx = x2-x1
        dy = y2-y1
        if dx != 0:
            if dx>0:
                if dy!=0:
                    if dy>0:
                        if dx>=dy: #premier octan
                            e = dx ; dx = 2*e; dy *= 2
                            for x1 in range (x1,x2+1):
                                l+=[(x1,y1)]
                                e-= dy
                                if e<0 :
                                    y1+=1
                                    e+=dx
                        else : #second octan
                            e = dy ; dy = 2*e ; dx*=2
                            for y1 in range (y1,y2+1):
                                e-= dx
                                l+=[(x1,y1)]
                                if e<0 :
                                    x1+=1
                                    e+=dy
                    else : #huitième octan
                        if dx>= -dy:
                            e = dx ; dx = 2*e; dy *= 2
                            for x1 in range (x1,x2+1):
                                l+=[(x1,y1)]
                                e+= dy
                                if e<0 :
                                    y1-=1
                                    e+=dx
                        else : #septième octan
                            e = dy ; dy = 2*e; dx *= 2
                            for y1 in range (y1,y2-1,-1):
                                l+=[(x1,y1)]
                                e+= dx
                                if e>0 :
                                    x1+=1
                                    e+=dy
                else : #horizontal
                    for x1 in range(x1,x2+1):
                            l += [(x1,y1)]
            else :
                if dy!=0 :
                    if dy>0:
                        if -dx>=dy : #quatrième octan
                            e = dx ; dx = 2*e; dy *= 2
                            for x1 in range (x1,x2-1,-1):
                                l+=[(x1,y1)]
                                e+= dy
                                if e>=0 :
                                    y1+=1
                                    e+=dx
                        else : #troisième octan
                            e = dy ; dy = 2*e; dx *= 2
                            for y1 in range (y1,y2+1):
                                l+=[(x1,y1)]
                                e+= dx
                                if e<=0 :
                                    x1-=1
                                    e+=dy
                    else :
                        if dx<=dy : #cinquième octan
                            e = dx ; dx = 2*e; dy *= 2
                            for x1 in range (x1,x2-1,-1):
                                l+=[(x1,y1)]
                                e-= dy
                                if e>=0 :
                                    y1-=1
                                    e+=dx
                        else : #sixième octan
                            e = dy ; dy *= 2; dx *= 2
                            for y1 in range (y1,y2-1,-1):
                                l+=[(x1,y1)]
                                e-= dx
                                if e>=0 :
                                    x1-=1
                                    e+=dy
                else : #horizontal
                    for x1 in range(x1,x2-1,-1):
                        l += [(x1,y1)]
        else :
            if dy!=0 :
                if dy>0 : #vertical vers le haut
                    for y1 in range(y1,y2+1):
                        l += [(x1,y1)]
                else :#vertical vers le bas
                    for y1 in range(y1,y2-1,-1):
                        l += [(x1,y1)]
            else : #chemin de A à A ( = A par convention )
                l+=[(x1,y1)]

        return(l)



    # "Calcul" des positions des cases formant le périmètre de l'enveloppe convexe



    def evp_cvx(l) :
        """
        Calcul la bordure exterieure de l'enveloppe convexe de l
        une liste de points extrémaux
        """

        if l == [] :
            return l
        pos = []
        m0 = len(l)
        if m0>=2 :
            l+= [l[0],l[1]]
            for k in range(1,m0+1):
                x,y = l[k] ; x1,y1 = l[k+1];x0,y0 = l[k-1]
                dx = x1-x ; dy = y1-y;dx0 = x-x0;dy0 = y-y0
                if dx != 0:
                    if dx>0:
                        if dy !=0:
                            if dy>0 : #wrap by the bottom
                                if dx0<=0 :
                                    pos+=[(x-1,y)]
                                if dy0<=0 :
                                    pos+= [(x,y-1)]
                                if dx>=dy :
                                    pos+=Calcul_Convexe.bresenham((x+1,y),(x1,y1-1))
                                else :
                                    pos+=Calcul_Convexe.bresenham((x+1,y),(x1+1,y1))
                            else :
                                if dy0>=0 :
                                    pos+=[(x,y+1)]
                                if dx0<=0:
                                    pos+=[(x-1,y)]
                                if dx>=-dy :
                                    pos += Calcul_Convexe.bresenham((x,y-1),(x1,y1-1))
                                else :
                                    pos+= Calcul_Convexe.bresenham((x,y-1),(x1-1,y1))
                        else :
                            if dx0<=0 :
                                pos+=[(x-1,y)]
                            pos+=Calcul_Convexe.bresenham((x,y-1),(x1,y1-1))
                    else : #wrap by the top
                        if dy !=0:
                            if dy>0 :
                                if dy0<=0 :
                                    pos+=[(x,y-1)]
                                if dx0>=0 :
                                    pos+=[(x+1,y)]
                                if -dx>=dy :

                                    pos += Calcul_Convexe.bresenham((x,y+1),(x1,y1+1))
                                else :
                                    pos+=Calcul_Convexe.bresenham((x,y+1),(x1+1,y1))
                            else :
                                if dx0>=0 :
                                    pos+=[(x+1,y)]
                                if dy0>=0:
                                    pos+=[(x,y+1)]
                                if dx<=dy :#comme -dx>=-dy
                                    pos += Calcul_Convexe.bresenham((x-1,y),(x1,y1+1))
                                else :
                                    pos+=Calcul_Convexe.bresenham((x-1,y),(x1-1,y1))
                        else :
                            if dx0>=0 :
                                pos+=[(x+1,y)]
                            pos+=Calcul_Convexe.bresenham((x,y+1),(x1,y1+1))
                else :
                    if dy<0 :
                        if dy0>=0 :
                            pos+=[(x,y+1)]
                        pos += Calcul_Convexe.bresenham((x-1,y),(x1-1,y1))
                    else :
                        if dy0<=0 :
                            pos+=[(x,y-1)]
                        pos += Calcul_Convexe.bresenham((x+1,y),(x1+1,y1))



            for k in range(2): #retirer les 2 ajoutés au début
                l.pop()


            #retirer les doublons :
            new_list = []
            if pos[0] == pos[-1]:
                pos.pop()

            #si doublon il y a, alors ils se suivent (par construction)
            for k in range(len(pos)-2) :
                if pos[-2] != pos[-1] :
                    new_list.append(pos[-1])
                pos.pop()

            #gère le cas des 2 derniers :
            for i in range(2) :
                val = pos[1-i]
                if val not in new_list :
                    new_list.append(val)

        else: #si un seul point le périmètre de l'enveloppe convexe est trivial
            x,y = l[0]
            new_list = [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

        return new_list





## Un Environment S'Accompagnant D'Un Affichage

class Environment_Z2 (object) :

    def __init__(self, grid_length = 10, nb_init = 1, advantage = 3, n_infections_per_step = 1, proba = 0.5, nb_vaccin = 2) :
        self.n = grid_length
        self.grid = np.zeros((self.n,self.n), dtype = np.int8)
        self.infected = []
        self.p = proba
        self.nb_vaccin = nb_vaccin
        self.advantage = advantage
        self.nb_init = nb_init
        self.n_steps = n_infections_per_step
        self.n_infected = nb_init



    def spread(self) :
        """
        Fait se propager le virus sur la grille
        """

        temp = []
        for x,y in self.infected :
            for i in [-1,1]:
                (x1,y1) = (x+i,y) ; (x2,y2) = (x,y+i)

                if self.exist(x1,y1) and self.grid[x1,y1] == 0 and np.random.random() < self.p :
                    self.grid[x1,y1] = 1
                    temp.append([x1,y1])
                    self.n_infected+=1

                if self.exist(x2,y2) and self.grid[x2,y2] == 0 and np.random.random() < self.p :
                    self.grid[x2,y2] = 1
                    temp.append([x2,y2])
                    self.n_infected+=1

        self.infected+=temp
        self.reduce_infected()
        return temp



    def von_neumann(self,x,y) :
        """
        Retourne True si l'une des cases dans le voisinage
        de Van Neumann du point (x,y) est susceptible.
        """

        for i in [-1,1]:
            if self.exist(x+i,y) and self.grid[(x+i,y)] == 0 or self.exist(x,y+i) and  self.grid[(x,y+i)] == 0 :
                return True
        return False



    def reduce_infected(self) :
        """
        Permet d'avoir à traiter uniquement une
        liste "courte" et de réduire le nombre
        d'opérations inutiles.
        """

        infected_ = []
        for pos in self.infected :
            if self.von_neumann(*pos) :
                infected_.append(pos)
        self.infected = infected_



    def exist(self,x,y) :
        """
        Renvoie True si le point (x,y) est dans la grille n*n
        """

        if (0 <= x < self.n) and (0 <= y < self.n) :
            return True
        return False


    def is_finished(self) :
        """
        Renvoie True si le virus ne peut plus se propager
        """

        for x,y in self.infected :
            for i in [-1,1]:
                if (self.exist(x+i,y) and self.grid[x+i,y] == 0) or (self.exist(x,y+i) and self.grid[x,y+i] == 0) :
                    return False
        return True



    def clear(self) :
        """
        Efface les données de la grille
        """

        self.grid = np.zeros((self.n,self.n), dtype = np.int8)
        self.infected = []
        self.n_infected = self.nb_init



    def start(self) :
        """
        (re)Initialise la grille et place les premiers infectés
        """

        for _ in range(self.nb_init) :
            if self.nb_init > self.n**self.n :
                raise ("Trop d'infectés initiaux")
            x,y = np.random.randint(0,self.n), np.random.randint(0,self.n)
            while self.grid[x,y] == 1 :
                x,y = np.random.randint(0,self.n), np.random.randint(0,self.n)
            self.grid[x,y] = 1
            self.infected.append([x,y])


        for _ in range(self.advantage) :
            temp = []
            for x,y in self.infected :
                for i in [-1,1]:
                    (x1,y1) = (x+i,y) ; (x2,y2) = (x,y+i)

                    if self.exist(x1,y1) and self.grid[x1,y1] == 0 and np.random.random() < self.p :
                        self.grid[x1,y1] = 1
                        temp.append([x1,y1])
                        self.n_infected+=1

                    if self.exist(x2,y2) and self.grid[x2,y2] == 0 and np.random.random() < self.p :
                        self.grid[x2,y2] = 1
                        temp.append([x2,y2])
                        self.n_infected+=1

            self.infected += temp
        return self.infected



    def step(self, x, y) :
        """
        Vaccine, quand c'est possible, le point (x,y) du plan
        """
        try :
            if self.grid[self.n-1-y,x] == 0 :
                self.grid[self.n-1-y,x] = 2

        except :
            pass



    def initiate(self,grille) :
        """
        Construit un environnement à partir d'une grille donnée
        """

        self.grid = np.zeros((self.n,self.n), dtype = np.int8)
        self.n = len(grille)
        self.infected = []
        for i in range(self.n) :
            for j in range(self.n) :
                if grille[i,j] == 1 :
                    self.grid[i,j] = 1
                    self.infected.append([i,j])
                if grille[i,j] == 2 :
                    self.grid[i,j] = 2
        self.n_infected = len(self.infected)




    ## Affichage
    def disp(self) :
        """
        Affiche le jeu dans la console
        """

        for line in self.grid :
            print (line)
        print('\n')


    def play(self) :
        """
        Affiche le jeu, cliquer sur la case à vacciner
        """


        self.clear()
        self.start()

        b = False
        tour = 0


        def save_fig(x) :
            plt.savefig(format = 'jpg', fname = f'Exemple de propagation d\'un virus au tour {tour} pour une probabilité {self.p}.jpg')




        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]}, figsize = (10, 8))
        fig.subplots_adjust(left = 0, bottom = 0,
                            right = 1, top = 0.95, wspace = 0, hspace = 0)


        ax[0].set_title('An infection is spreading, save as many people as possible !')


        posbutton = plt.axes([0.81, 0.01, 0.1, 0.06])
        button = Button(posbutton, 'Save', color='.90', hovercolor='1')
        cid_butt = button.on_clicked(save_fig)



        ax[0].set_xlim(0,self.n)
        ax[0].set_ylim(0,self.n)

        ax[1].set_xlim(-2,-1)
        ax[1].set_ylim(-2,-1)


        ax[1].add_patch(patches.Rectangle(
                                [-3,-3], 0, 0,
                                facecolor = '#FF5733',
                                fill = True, hatch = 'x',
                                linewidth = 1,label = 'Infected')
                                )
        ax[1].add_patch(patches.Rectangle(
                                [-3,-3],0,0,
                                facecolor = '#3CE11A',
                                fill = True, hatch = 'x',
                                linewidth = 1,label = 'Vaccinated')
                                )
        ax[1].legend(loc='center left')


        ax[0].set_xticks(list(range(self.n)))
        ax[0].set_yticks(list(range(self.n)))
        ax[0].grid()

        for p in self.infected :
            ax[0].add_artist(
                patches.Rectangle(
                                [p[1],self.n-1-p[0]], 1, 1,
                                facecolor = '#FF5733',
                                fill = True, hatch = 'x',
                                linewidth = 1,)
                                )

        def tellme(s) :
            ax[0].set_title(s)
            plt.draw()



        while not self.is_finished():
            tour+=1

            pts = np.asarray(plt.ginput(self.nb_vaccin, timeout=-1))

            while np.any(pts<0) :
                pts = np.asarray(plt.ginput(self.nb_vaccin, timeout=-1))


            if len(pts) != self.nb_vaccin :
                b = True
                break

            for k in range(self.nb_vaccin) :

                x,y = int(pts[k][0]), int(pts[k][1])
                if self.grid[self.n-1-y, x] == 0 :
                    ax[0].add_artist(
                        patches.Rectangle([x,y], 1, 1,
                                        facecolor = '#3CE11A',
                                        fill = True, hatch = 'x',
                                        linewidth = 1)
                                        )
                    self.grid[self.n-1-y, x] = 2


            tellme('An infection is spreading, save as many people as possible !')


            plt.waitforbuttonpress()
            for _ in range(self.n_steps) :
                for p in self.spread() :
                    tellme('An infection is spreading, save as many people as possible !')
                    ax[0].add_artist(
                        patches.Rectangle(
                                        [p[1],self.n-1-p[0]], 1, 1,
                                        facecolor = '#FF5733',
                                        fill = True, hatch = 'x',
                                        linewidth = 1))


        if b :
            plt.close('all')
        else :

            ax[0].text(self.n/2,self.n/2,"Félicitation" if self.n_infected<=self.n*self.n/2 else "Dommage" , horizontalalignment = 'center', verticalalignment = 'center', color = 'yellow', fontsize = 60, alpha = 0.9, bbox=dict(facecolor='red', alpha=0.7,boxstyle = 'round4'))

            tellme(f'résolu en {tour} tours avec {self.n**2-self.n_infected} personnes sauvés')


            plt.waitforbuttonpress()
            plt.waitforbuttonpress()
            plt.close('all')



    def jeu_aide_IA(self,IA) :
        """
        Affiche le jeu en faisant jouer une IA
        """

        self.clear()
        self.start()


        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]}, figsize = (10, 8))
        fig.subplots_adjust(left = 0, bottom = 0,
                            right = 1, top = 0.95, wspace = 0, hspace = 0)


        ax[0].set_title('An infection is spreading, save as many people as possible !')


        def vaccinate(a) :
            (x,y) = IA(self.grid)

            if self.exist(self.n-1-y, x) and self.grid[self.n-1-y, x] == 0 :
                ax[0].add_artist(
                    patches.Rectangle([x,y], 1, 1,
                                    facecolor = '#3CE11A',
                                    fill = True, hatch = 'x',
                                    linewidth = 1)
                                    )
                self.grid[self.n-1-y, x] = 2
            plt.show()

        def infect(a) :
            for p in self.spread() :
                ax[0].add_artist(
                    patches.Rectangle(
                                    [p[1],self.n-1-p[0]], 1, 1,
                                    facecolor = '#FF5733',
                                    fill = True, hatch = 'x',
                                    linewidth = 1))
            plt.show()





        posbutton = plt.axes([0.81, 0.01, 0.1, 0.06])
        button = Button(posbutton, 'Vaccinate', color='.90', hovercolor='1')
        cid_butt = button.on_clicked(vaccinate)

        posbutton2 = plt.axes([0.70, 0.01, 0.1, 0.06])
        button2 = Button(posbutton2, 'Infect', color='.90', hovercolor='1')
        button2.on_clicked(infect)

        def save_fig(x) :
            plt.savefig(format = 'jpg', fname = f'Exemple de propagation d\'un virus empêché par une IA.jpg')

        posbutton3 = plt.axes([0.59, 0.01, 0.1, 0.06])
        button3 = Button(posbutton3, 'Save', color='.90', hovercolor='1')
        button3.on_clicked(save_fig)

        def restart(x) :
            plt.close('all')
            self.jeu_aide_IA(IA)

        posbutton4 = plt.axes([0.48, 0.01, 0.1, 0.06])
        button4 = Button(posbutton4, 'Restart', color='.90', hovercolor='1')
        button4.on_clicked(restart)


        ax[0].set_xlim(0,self.n)
        ax[0].set_ylim(0,self.n)



        ax[1].add_patch(patches.Rectangle(
                                [-3,-3], 0, 0,
                                facecolor = '#FF5733',
                                fill = True, hatch = 'x',
                                linewidth = 1,label = 'Infected')
                                )
        ax[1].add_patch(patches.Rectangle(
                                [-3,-3],0,0,
                                facecolor = '#3CE11A',
                                fill = True, hatch = 'x',
                                linewidth = 1,label = 'Vaccinated')
                                )
        ax[1].legend(loc='center left')


        ax[0].set_xticks(list(range(self.n)))
        ax[0].set_yticks(list(range(self.n)))
        ax[0].grid()

        for p in self.infected :
            ax[0].add_artist(
                patches.Rectangle(
                                [p[1],self.n-1-p[0]], 1, 1,
                                facecolor = '#FF5733',
                                fill = True, hatch = 'x',
                                linewidth = 1,)
                                )



            plt.show()
        plt.show()






## Les Différentes IAs

class IAs(object) :

    # Transformation du plan

    def mat_to_points (M):
        """ Traduit d'une matrice à un plan : ce que lit
        le programme
        """

        l = []
        n = len(M)
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    l+=[(j,n-1-i)]
        return l


    def points_to_mat(t,n):
        """
        Traduit d'un plan à une matrice
        """

        m = len(t)
        t2 = []
        for k in range(m):
            t2+= [(n-1-t[k][1],t[k][0])]
        return(t2)


    # Choix du premier point

    def cadre (l,n):
        """
        Place le virus dans un quartile du graphe : le
        choix initial optimal est alors à l'opposé de cette
        position dominante
        """

        maxx, maxy, minx, miny = 0,0,n,n
        for x,y in l :
            if x>maxx :
                maxx = x
            if y>maxy :
                maxy=y
            if x<minx :
                minx=x
            if y<miny :
                miny=y

        return(minx,maxx,miny,maxy)

    def chx(pos,t,n):
        """
        Choisi la meilleure position en fonction des
        positions du virus
        """

        if pos == 'hd':
            return IAs.closest(t, lambda x,y : x*x + y*y )

        if pos == 'hg':
            return IAs.closest(t, lambda x,y : (n-x-1)**2 + y*y )

        if pos == 'bg':
            return IAs.closest(t, lambda x,y : (n-x-1)**2 + (n-y-1)**2 )

        if pos == 'bd':
            return IAs.closest(t, lambda x,y : x*x + (n-y-1)**2 )




    def positions(grille):
        """
        Position translatée dans le plan des infectés
        et vaccinés (comme mat_to_point)
        """

        n = len(grille)
        vaccin,infect = [],[]
        for i in range(n):
            for j in range(n):
                if grille[i,j]==1 :
                    infect.append((j,n-1-i))
                if grille[i,j]==2:
                    vaccin.append((j,n-1-i))
        return vaccin,infect


    # voisinages et fonctions utiles

    def Moore (pos):
        """
        Voisinage de Moore
        """

        x,y = pos
        return [(x+1,y),(x,y+1),(x-1,y),(x,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1)]

    def Von_Neumann (pos):
        """
        Voisinage de Van Neumann
        """

        x,y = pos
        return [(x+1,y),(x,y+1),(x-1,y),(x,y-1)]

    def intersection(lst1, lst2):
        """
        Renvoie l'intersection de deux listes
        """

        lst3 = [value for value in lst1 if value in lst2]
        return lst3


    def closest(l, norm) :
        """
        Renvoie l'élément de l le plus petit pour une
        norme donnée
        """

        if len(l) == 0 :
            raise IndexError
        temp = l[0]
        for val in l :
            if norm(*val) < norm(*temp) :
                temp = val
        return temp



    # Politiques

    def exist(y,x,n) :
        """
        Vérifie que le point (x,y) est bien dans la
        grille observé
        """

        if (0 <= x < n) and (0 <= y < n) :
            return True
        return False



    def barycentre(infected) :
        """
        Calcule le barycentre de l'ensemble des infectés
        """

        barx,bary = 0,0
        for x,y in infected :
            barx += x
            bary += y
        return barx/len(infected), bary/len(infected)



    def policy_bar(A,B,infect,n) :
        """
        Politique de choix entre le point A et B
        """

        x1,y1 = A; x2,y2 = B
        if not IAs.exist(x1,y1,n) :
            return B
        if not IAs.exist(x2,y2,n) :
            return A

        bar = IAs.barycentre(infect)
        d1 = (A[0] - bar[0])**2 + (A[1] - bar[1])**2
        d2 = (B[0] - bar[0])**2 + (B[1] - bar[1])**2
        if d1<d2 :
            return A
        return B


    # Pour combler les trous une fois le virus infecté


    def ending(vaccin,grille) :
        """
        Chois, une fois l'enveloppe convexe complétée,
        un individus à vacciner en essayant d'empêcher le
        plus de contaminations possibles
        """

        n = len(grille)
        boul = False
        temp = []
        for x,y in vaccin :

            temp = [p for p in vaccin if p[1] == y]
            for i in range(min(temp)[0], max(temp)[0]) :
                if grille[n-1-y, i] == 0 :
                    return i,y


        for i in range(n) :
            for j in range(n) :
                if grille[i,j] == 0 :
                    return j,n-1-i

        return 0,0



    # Sélection des 2 choix possibles

    def select(cvx,infect,vaccin,grille) :
        """
        Selectionne des individus possiblement interessant
        à vacciner : les bouts de l'enveloppe convexe déjà
        construits
        """

        rez = []
        n = len(cvx)
        for k in range(n) :
            v = IAs.intersection(IAs.Moore(cvx[k])+[cvx[k]],vaccin)
            if len(v)>0 :
                if len(IAs.intersection(IAs.Moore(cvx[(k-1)%n])+[cvx[(k-1)%n]],vaccin))== 0 :
                    rez.append(cvx[k])
                if len(IAs.intersection(IAs.Moore(cvx[(k+1)%n])+[cvx[(k+1)%n]],vaccin))== 0 :
                    rez.append(cvx[k])
        return list(set(rez))



    def epure(possible, infect) :
        """
        Supprime les élements litigieux dans le choix
        des bouts du lacet formant l'enveloppe convexe
        """
        m = len(possible)
        if m <= 2 :
            return possible

        if m > 2 :

            norm = lambda a,b,c,d : (a-c)**2+(b-d)**2
            p = len(infect)
            val = np.zeros(m,dtype = np.int8)

            for k in range(m) :
                val[k] = norm(*possible[k],*infect[0])
                for j in range(1,p) :
                    val[k] = min(val[k], norm(*possible[k],*infect[j]))

            m1 = np.inf ; m2 = np.inf
            for x in val:
                if x <= m1:
                    m1, m2 = x, m1
                elif x < m2:
                    m2 = x

            temp1,temp2 = -1,-1
            for k in range(m) :
                if val[k] == m1 and temp1 < 0 :
                    temp1 = k
                if val[k] == m2 and temp1 != k :
                    temp2 = k
            return[possible[temp1],possible[temp2]]



    ## IAs CONVEXE

    def IA_Cvx_Bary(grille):
        """ Renvoie la coordonnée (/!\dans le plan) du point
        le plus avantageux à vacciner suivant la politique
        du Barycentre : le plus proche du barycentre
        des infectés
        """

        n = len(grille)
        vaccin,infect = IAs.positions(grille)
        t0  = Calcul_Convexe.graham(infect)
        cvx = Calcul_Convexe.evp_cvx(t0)
        if vaccin == []:
            xmin,xmax,ymin,ymax = IAs.cadre(t0,n)
            if n-xmax-1>xmin : #plutôt à gauche
                if n-ymax-1>ymin: #plutôt en bas
                    (x1,y1) = IAs.chx('bg',cvx,n)
                else :
                    (x1,y1) = IAs.chx('hg',cvx,n)
            else :
                if n-ymax-1>ymin:
                    (x1,y1) = IAs.chx('bd',cvx,n)
                else :
                    (x1,y1) = IAs.chx('hd',cvx,n)

            return (x1,y1)
        else :
            possible = IAs.select(cvx,infect,vaccin,grille)
            n_possible = len(possible)
            if n_possible == 2 :
                choix1,choix2 = possible
                if not (IAs.exist(*choix1,n) or IAs.exist(*choix2,n)) :
                    return IAs.ending(vaccin,grille)

                return IAs.policy_bar(choix1,choix2,infect,n)

            elif n_possible > 2 :
                a,b = IAs.barycentre(infect)
                x1,y1 = IAs.closest(possible, lambda x,y : -(a-x)**2 - (b-y)**2)
                while len(possible)>1 and not IAs.exist(x1,y1,n)  :
                    possible.remove((x1,y1))
                    x1,y1 = IAs.closest(possible, lambda x,y : -(a-x)**2 - (b-y)**2)
                if len(possible) == 0 :
                    return IAs.ending(vaccin,grille)
                return x1,y1
            elif n_possible == 1 :
                return possible[0]

            else :
                for p in cvx :
                    if IAs.exist(*p,n) and grille[n-1-p[1],p[0]] == 0 :
                        for neigh in IAs.Moore(p) :
                            if IAs.exist(*neigh,n) :
                                if grille[n-neigh[1]-1,neigh[0]] == 1 :
                                    return p

                return IAs.ending(vaccin,grille)




    # Avec Monte Carlo ?


    def valuation(env,y) :
        """
        Calcul l'espérance du nombre de personnes sauvées
        """

        if env.is_finished() :
            return env.n*env.n - env.n_infected


        vaccin,infect = IAs.positions(env.grid)
        t0  = Calcul_Convexe.graham(infect)
        cvx = Calcul_Convexe.evp_cvx(t0)
        possible = IAs.select(cvx,infect,vaccin,env.grid)
        possible = IAs.epure(possible,infect)
        if len(possible) in [1,2] :
            env.step(*possible[np.random.randint(len(possible))])
            env.spread()
            return y*valuation(env,y)

        return env.n*env.n - env.n_infected







    def IA_Cvx_combinaison(env,mu = 10):
        """
        Renvoie la coordonnée (/!\dans le plan) du point le plus
        avantageux à vacciner
        choisit d'abord la politique Barycentre puis quand on
        est assez proche de la fin (a mu près) change pour
        Monte Carlo
        """
        """
        Politique Monte Carlo : calcul l'espérance attendue
        du nombre d'individus infectés au final puis
        vaccine celui avec le moins
        """

        vaccin,infect = IAs.positions(env.grid)
        t0  = Calcul_Convexe.graham(infect)
        cvx = Calcul_Convexe.evp_cvx(t0)
        if vaccin == []:
            xmin,xmax,ymin,ymax = IAs.cadre(t0,env.n)
            if env.n-xmax-1>xmin : #plutôt à gauche
                if env.n-ymax-1>ymin: #plutôt en bas
                    (x1,y1) = IAs.chx('bg',cvx,env.n)
                else :
                    (x1,y1) = IAs.chx('hg',cvx,env.n)
            else :
                if env.n-ymax-1>ymin:
                    (x1,y1) = IAs.chx('bd',cvx,env.n)
                else :
                    (x1,y1) = IAs.chx('hd',cvx,env.n)

            return (x1,y1)
        else :
            possible = IAs.select(cvx,infect,vaccin,env.grid)
            possible = IAs.epure(possible, infect)
            n_possible = len(possible)

            if n_possible > 1 and env.n*len(list(filter(lambda a : IAs.exist(a[0],a[1],env.n) and env.grid[a[1],env.n-1-a[0]] == 0,cvx)))/len(cvx) < mu :

                if not(IAs.exist(*possible[0],env.n) or IAs.exist(*possible[1],env.n)) :
                    return IAs.ending(vaccin,env.grid)

                if not IAs.exist(*possible[0],env.n) :
                    return possible[1]

                if not IAs.exist(*possible[1],env.n) :
                    return possible[0]


                env2 = Environment_Z2(grid_length = env.n, proba = env.p, n_infections_per_step = env.n_steps, nb_vaccin = env.nb_vaccin)
                env1 = Environment_Z2(grid_length = env.n, proba = env.p, n_infections_per_step = env.n_steps, nb_vaccin = env.nb_vaccin)

                rez1 = 0
                rez2 = 0
                n_eval = env.n
                for _ in range(n_eval) :
                    env1.initiate(env.grid)
                    env2.initiate(env.grid)
                    env1.step(*possible[0])
                    env2.step(*possible[1])
                    rez1+=valuation(env1,0.99)
                    rez2+=valuation(env2,0.99)

                return possible[0] if rez1>rez2 else possible[1]
            else :
                if n_possible == 2 :
                    choix1,choix2 = possible
                    if not (IAs.exist(*choix1,env.n) or exist(*choix2,env.n)) :
                        return IAs.ending(vaccin,env.grid)

                    return IAs.policy_bar(choix1,choix2,infect,env.n)

                    return x1,y1

                elif n_possible == 1 :
                    return possible[0]

                else :
                    for p in cvx :
                        if IAs.exist(*p,n) and grille[n-1-p[1],p[0]] == 0 :
                            for neigh in IAs.Moore(p) :
                                if IAs.exist(*neigh,n) :
                                    if grille[n-neigh[1]-1,neigh[0]] == 1 :
                                        return p

                    return IAs.ending(vaccin,grille)





## Pour tester l'algorithme

"""

Créez un environement avec, par exemple, la commande :

env = Environment_Z2(grid_length = 30, nb_init = 1, advantage = 3, n_infections_per_step = 1, proba = 0.5, nb_vaccin = 2)

qui considère une grille de 30*30 avec un seul infécté initial (pas encore présent sur la grille) et qui laissera à l'initialisation le virus se propager 3 tours avant de pouvoir commencer à vacciner. On peut vacciner 2 personnes par tour puis l'infection se propage "une fois" avec une probabilité de 50%.

Pour l'affichage du jeu avec possibilité d'utiliser une IA, tapez env.jeu_aide_IA(IA_Cvx) dans la console et executez.

Les boutons, de droite à gauche, vous permettent de :
-vacciner une case suivant l'algorithme donné en entré,
-infecter pour un tour suivant la probabilité donné à la création de l'environment,
-sauvegarder l'image présente en face de vous (/!\ en sauvegarder une nouvelle écrasera la précédente, changez le nom de sauvegarde à la ligne 312 de Jeu_Z2 ou celui de l'image déjà téléchargée pour en ajouter une nouvelle),
-recommencer une nouvelle partie si celle en cours vous ennuie :) (pour quitter il suffit de fermer la page...)


Voici un exemple de test de probabilité à faire pour tester l'efficacité de l'algorithme et comparer différentes politiques de choix :

env = Environment_Z2(grid_length = 10, nb_init = 1, advantage = 3, n_infections_per_step = 1, proba = 0.5, nb_vaccin = 2)

n_eval = 1000 ; n_vaccin = 2

tot = 0
mean_n_infect_ini = 0
IA = Monte_Carlo_Graham # IA_Cvx_combinaison
#IA_Cvx (/!\ à mettre avec IA(env.grid))
# Pensez à les importer avant...

for _ in range(n_eval) :
    env.clear()
    env.start()
    mean_n_infect_ini += env.n_infected
    while not env.is_finished() :
        for _ in range(n_vaccin) :
            (x,y) = IA(env)
            env.step(x,y)
        env.spread()
    tot+=env.n_infected



# Pour le pourcentage de personnes infectés :
tot/n_eval/100
# Quelques valeurs : grid_length=30 & proba=0.5 => 13% = 117 infectés
# grid_length=8 & proba=1 => 34% = 22 infectés en moyenne


#IA_Cvx_combinaison : grid_length=8 & proba=0.5 =>
tot/n_eval/64 -> 0.30234375 -> 19.35 infectés en moyenne
grid_length = 10 => 0.26778 infectés

# Monte_Carlo_Graham : grid_length=8 & proba=0.5 =>
tot/n_eval/64 -> 0.3078125 -> 19.7 infectés en moyenne
0.27226 pour grid_length = 10

# Pour le nombre d'inféctés initiaux
mean_n_infect_ini/n_eval
# idem => 10 infectés initiaux

"""
