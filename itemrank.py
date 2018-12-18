import numpy as np
import sys


def main(argv):
    """
    Méthode qui récupère en argument le nom du fichier .csv et si c'est le cas la vecteur de personnalisation.
    Il appelle ensuite la méthode itemRank qui retournera le score final.
    """
    np.set_printoptions(linewidth=np.inf)   #afficher les resultats complets

    if len(argv) == 2 or len(argv) == 3:
        path_file_graphe = argv[0]  # Récupère le chemin du fichier placé en argument
        path_file_vp = argv[1]  # Récupère le chemin du fichier placé en argument
        A = np.loadtxt(path_file_graphe, delimiter=",", dtype=np.int)  # Converti le fichier csv en array
        v = np.loadtxt(path_file_vp, delimiter=",", dtype=np.int)  # Converti le fichier csv en array

        A = np.asmatrix(A)  # Converti np.array en np.matrix
        v = np.asmatrix(v)

        # Test si la matrice lue est carrée et si le vecteur de personnalisation a la meme quantité de colonnes
        if A.size != A.shape[0] ** 2 and v.size != A.shape[0]:
            print("ERROR : Verifier que la matrice est carré ou le nombre de colonnes du vecteur de personnalisation !")
            return

        if len(argv) == 3 and argv[2] == "1":
            xt = itemRank(A, 0.15, v, True)
        else:
            xt = itemRank(A, 0.15, v, False)

        if xt is None:
            print("Erreur dans le calcul !")
            return
        print("Score ItemRank final :")
        print(np.squeeze(xt))

    else:
        print("utilisez le programme avec des arguments:")
        print("itemrank.py arg1 arg2 méthode")
        print("arg1: fichier contenant la matrice d'adjacence du graphe")
        print("arg2: fichier contenant le vecteur de personnalisation")
        print("méthode: cette valeur est optionnelle, <1> pour utiliser l'inversion matricielle sinon par défaut <0> la "
              "méthode par recurrence")


def recurrence(alpha, P, xtu, vu, i=1):
    """
    Méthode par recurrence permettant le calcul recursive du vecteur ItemRank
    ce calcul s'arrête jusqu'à ce que la différence entre la valeur précédente et celle calculée convergent
    """
    xtuNew = alpha * P.transpose() * xtu + (1 - alpha) * vu

    if i == 1 or i == 2 or i == 3:
        print("Itération :", i)
        print(np.squeeze(np.asarray(xtuNew)))

    if (np.linalg.norm(np.abs(xtu - xtuNew), 1)) < 10 ** -8:
        print ("Nombre total d'itérations: ", i)
        return xtuNew
    xtuNew = recurrence(alpha, P, xtuNew, vu, i + 1)
    return xtuNew


def itemRank(A, alpha=0.15, v=None, m=False):
    """
    Méthode calculant à partir de la matrice d'adjacence, d'un paramètre de téléportation et d'un vecteur de personnalisation
    le Score ItemRank, le paramètre m indique la méthode de resolution choisi par l'utilisateur
    """

    print("Matrice d'adjacence :")
    print(A)

    Aout = np.sum(A, axis=1)  # Vecteur des degrés sortant
    Ain = np.sum(A, axis=0)  # Vecteur des degrés entrant
    print("Vecteur des degrés entrant :")
    print(Ain)

    print("Vecteur des degrés sortant :")
    print(Aout)

    vu = (v / np.sum(v))
    vu = vu.transpose()
    print("Vecteur de personalisation :")
    print(vu)

    # Calcul de la matrice de probabilités de transition
    P = A / Aout[:]
    print("Matrice de probabilités de transition :")
    print(P)

    if m:
        print("Calcul par inversion matricielle")
        I = np.asmatrix(np.identity(A.shape[0]))
        res = (1 - alpha) * (I - alpha * P.transpose()) ** -1 * vu
    else:
        print("Calcul par recurrence")
        res = recurrence(alpha, P, vu, vu, 1)

    return np.asarray(res)


if __name__ == '__main__':
    main(sys.argv[1:])
