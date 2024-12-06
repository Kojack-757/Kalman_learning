import numpy as np

def CalculKalman(F,X_init, P_init, B, Q, R,H, u,z,dt):

    if np.isscalar(dt):
        dt = dt * np.ones(len(z))
    X = [X_init]
    P = [P_init]
    K = []




    for i in range(len(z)):
        #F = np.array([[1, dt[i]], [0, 1]])
        print(f"F: \n {F} \n")

        X_pred = F @ X[i] + B * u[i]  #ATTENTION NE MARCHERA PLUS EN MATRICIELLE
        P_pred = F @ P[i] @ F.T + Q
        #RAJOUTER UNE VARIABLE AVEC TOUT LES PRED
        print('## -- Début du cycle -- ##')
        print("Prédiction:")
        print(f"X_pred: \n{X_pred}")
        print(f"P_pred: \n{P_pred}\n")

        #Calcul de l'innovation:

        y = z[i] - H @ X_pred           #innovaiton
        S = H @ P_pred @ H.T + R        #Covariance de l'innovation
        print("Innovation:")
        print(f"Innvation: \n{y}\n")
        print(f"Covar Innvation: \n{S}\n")


        K_gain = (P_pred @ H.T) / S   # ATTENTION FAIRE np.linalg.inv(S) quand on sera en Matricielle
        K.append(K_gain)

        print("Mise à jours gain Kalman:")
        print(f"K_gain: \n{K_gain}\n")

        print("Mesure:")
        print(f"z[i]: \n{z[i]}\n")

        X_update = X_pred + K_gain * y #ATTENTION NE MARCHERA PLUS EN MATRICIELLE
        P_update = (np.eye(2) - K_gain @ H) @ P_pred

        print("Mise à jour X et P")
        print(f"X_update: \n{X_update}")
        print(f"P_update: \n{P_update}\n")

        X.append(X_update)
        P.append(P_update)

    return np.array([X]), P
