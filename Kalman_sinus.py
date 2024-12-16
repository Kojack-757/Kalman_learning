
import numpy as np
import matplotlib.pyplot as plt
import Kalman_Perso
from scipy.interpolate import interp1d
plt.close('all')


#--- Signaux théorique ---##
dt = 1e-3     # time step
T  = 1       # total time
N  = int(T/dt) # number of data

A = 1
f = 10
w = 2*np.pi*f
t = np.arange(0, T, dt) # time array


X_th = np.sin(w*t)
V_th =  w*np.cos(w*t)
A_th = -w**2*np.sin(w*t)


## --- Mesure Accélération --- ###

dt_acc = dt #3e-6   # nouveau pas de temps pour A_mes
T_acc  = T        # total time
t_acc = np.arange(0, T_acc, dt_acc)  # vecteur de temps pour A_mes
N_acc  = len(t_acc) # number of data

interp_A_th = interp1d(t, A_th, kind='linear', fill_value="extrapolate")
A_th_acc = interp_A_th(t_acc)  # Accélération théorique échantillonnée toutes les 0.1s


amplitude_signal = np.max(np.abs(A_th_acc))
Err_Acc = amplitude_signal*1e-1
Var_Acc = Err_Acc**2

A_mes = A_th_acc + Err_Acc * np.random.randn(N_acc)


## --- Mesure Vitesse --- ###

dt_vit = dt #5e-5   # nouveau pas de temps pour A_mes
T_vit = T        # total time
t_vit = np.arange(0, T_vit, dt_vit)  # vecteur de temps pour A_mes
N_vit  = len(t_vit) # number of data

interp_V_th = interp1d(t, V_th, kind='linear', fill_value="extrapolate")
V_th_vit = interp_V_th(t_vit)  # Accélération théorique échantillonnée toutes les 0.1s

amplitude_signal = np.max(np.abs(V_th_vit))
Err_vit = amplitude_signal*1e-1
Var_vit = Err_vit**2

V_mes = V_th_vit + Err_vit*np.random.randn(N_vit)




## --- Kalman ---- #

F = np.array([[1, dt], [0, 1]])         # Modele de transition d'état
G = np.array([[dt**2/2],[dt]])          # Modele d'entrée
B = G                                   # Ca c'est plus pour m'y retrouver entre la litérature et Thomas
Q = np.array([Var_Acc])                 # Covariance de "processus" ici l'accélération
R = np.array([Var_vit])                 # Covariance de la mesure ici l'accélération
H = np.array([0, 1])

X_init = np.array([[0],[0]])
P_init = np.array([[1e5,0],[0, 1e10]])

X, P, Xpred,Ppred,Innov,CovarInnov = Kalman_Perso.CalculKalman(F=F,X_init=X_init,P_init=P_init,B=B,Q=Q,R=R,H=H,u=A_mes,z=V_mes,dt=dt)

X_pred = [state[0, 0] for state in Xpred]
X_pred = np.array(X_pred)

V_pred = [state[1, 0] for state in Xpred]
V_pred = np.array(V_pred)

X_filt = [state[0, 0] for state in X]
X_filt = np.array(X_filt)


V_filt = [state[1, 0] for state in X]
V_filt = np.array(V_filt)


fig,axs = plt.subplots(nrows=3,ncols=1)

axs[0].plot(t, X_th, label='Théorie', color='red')
axs[0].plot(t, X_filt[:-1], label='filtré', color='green')
axs[0].set_title('position')
axs[0].set_xlabel('Temps')
axs[0].legend()

axs[1].plot(t_vit, V_mes, label='Mesure', color='blue')
axs[1].plot(t, V_th, label='Théorie', color='red')
axs[1].plot(t, V_filt[:-1], label='filtré', color='green')
axs[1].set_title('Vitesse')
axs[1].set_xlabel('Temps')
axs[1].legend()


axs[2].plot(t_acc, A_mes, label='Mesure', color='blue')
axs[2].plot(t, A_th, label='Théorie', color='red')
axs[2].set_title('Accélération')
axs[2].set_xlabel('Temps')
axs[2].legend()

fig2,axs2 = plt.subplots(nrows=2,ncols=1)
axs2[0].plot(t, Innov, label='Innov', color='red')
axs2[0].set_title('Innovation')
axs2[0].set_xlabel('Temps')
axs2[0].legend()

axs2[1].plot(t, CovarInnov, label='CovarInnov', color='blue')
axs2[1].set_title('Covar Innovation')
axs2[1].set_xlabel('Temps')
axs2[1].legend()

fig3,axs3 = plt.subplots(nrows=2,ncols=1)
axs3[0].plot(t, X_th, label='Théorie', color='red')
axs3[0].plot(t, X_filt[:-1], label='filtré', color='green')
axs3[0].plot(t, X_pred, label='Prédiction', color='yellow')
axs3[0].set_title('Prediciton')
axs3[0].set_xlabel('Temps')
axs3[0].legend()

axs3[1].plot(t_vit, V_mes, label='Mesure', color='blue')
axs3[1].plot(t, V_th, label='Théorie', color='red')
axs3[1].plot(t, V_filt[:-1], label='filtré', color='green')
axs3[1].plot(t_vit, V_pred, label='Prediction', color='yellow')
axs3[1].set_title('Prediciton')
axs3[1].set_xlabel('Temps')
axs3[1].legend()

fig4,axs4 = plt.subplots(nrows=2,ncols=1)

X_diff = []
V_diff = []
for i in range(len(t)):
    X_diff.append(X_filt[i] - X_pred[i])
    V_diff.append(V_filt[i] - V_pred[i])

X_diff = np.array(X_diff)
V_diff = np.array(V_diff)

axs4[0].plot(t,X_diff, label='Position', color='black')
axs4[1].plot(t,V_diff, label='Vitesse', color='black')
axs4[1].set_title('Différence filtré-prédit')
axs4[1].set_xlabel('Temps')
axs4[1].legend()

plt.legend()
plt.show()
