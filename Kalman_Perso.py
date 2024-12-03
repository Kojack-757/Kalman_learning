import numpy as np

def CalculKalman(F,X_init, P_init, B, Q, R,H, u,y,dt):

    if np.isscalar(dt):
        dt = dt * np.ones(len(y))
    X = [X_init]
    P = [P_init]
    K = []




    for i in range(len(y)):
        #F = np.array([[1, dt[i]], [0, 1]])
        print(F)
        X_pred = F @ X[i] + B @ np.array([u[i]])
        P_pred = F @ P[i] @ F.T + Q

        print('## -- Début du cycle -- ##')
        print("Prédiction:")
        print(f"X_pred: \n{X_pred}")
        print(f"P_pred: \n{P_pred}\n")

        K_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        K.append(K_gain)

        print("Mise à jours gain Kalman:")
        print(f"K_gain: \n{K_gain}\n")

        print("Mesure:")
        print(f"z: \n{z}\n")

        X_update = X_pred + K_gain @ (z - H @ X_pred)
        P_update = (np.eye(2) - K_gain @ H) @ P_pred

        print("Mise à jour X et P")
        print(f"X_update: \n{X_update}")
        print(f"P_update: \n{P_update}\n")

        X.append(X_update)
        P.append(P_update)

    return np.array(X), np.array(P)
