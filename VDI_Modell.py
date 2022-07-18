import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 22})

E0_C = 1368
dtor = 2*np.pi/360
j = np.linspace(1, 365, 365)
j_h = np.reshape(np.tile(j, (24)), (24, 365)).T
hours = np.linspace(1, 24, 24)
x = 0.98556*j - 2.72
xh =  0.98556*j_h - 2.72
l_MEZ = 15
gamma_E = 30
alpha_E = 180
a = 0.506
b = 6.08
c = 1.6364
z = 270
s = 0.72
t = 3.2

ex = 1 + 0.0334*np.cos(0.9856*dtor*j - 2.72/360*2*np.pi)
E0 = E0_C*ex
delta = np.arcsin(0.3978*np.sin((xh - 77.51 + 1.92*np.sin(xh*dtor))*dtor))/dtor
ZGL = -7.66*np.sin(xh*dtor) - 9.87 * np.sin((2*xh + 24.99 + 3.83*np.sin(xh*dtor))*dtor)

def pos(lat= 48, lon=7):
    MOZ = hours - (l_MEZ - lon)*4/60
    WOZ = MOZ + ZGL/60
    omega = (WOZ - 12)*15
    N_data = pd.read_csv("stundenwerte_N_01443_19490101_20211231_hist/produkt_n_stunde_19490101_20211231_01443.txt", sep=";")
    N_data["MESS_DATUM"] = pd.to_datetime(N_data["MESS_DATUM"], format="%Y%m%d%H")
    print(N_data)

    gamma_s24 = np.arcsin(np.cos(omega*dtor) * np.cos(lat*dtor) * np.cos(delta*dtor) + np.sin(lat*dtor)*np.sin(delta*dtor))/dtor
    gamma_s = np.where(gamma_s24<0, 0, gamma_s24)
    alpha_s = np.where(WOZ <= 12, 180 - np.arccos((np.sin(lat*dtor)*np.sin(gamma_s*dtor) - np.sin(delta*dtor))/(np.cos(lat*dtor)*np.cos(gamma_s*dtor)))/dtor,
    180 + np.arccos((np.sin(lat*dtor)*np.sin(gamma_s*dtor) - np.sin(delta*dtor))/(np.cos(lat*dtor)*np.cos(gamma_s*dtor)))/dtor)

    theta_gen = np.arccos(np.sin(gamma_s*dtor)*np.cos(gamma_E*dtor) + np.cos(gamma_s*dtor)*np.sin(gamma_E*dtor)*np.cos((alpha_E - alpha_s)*dtor))/dtor
    m = np.where(gamma_s == 0, 0, 1/(np.sin(gamma_s*dtor) + a*(np.power(gamma_s + b, -c))))

    idx0 = np.where(gamma_s < 0.5)
    idx1 = np.where((gamma_s >= 0.5) & (gamma_s < 1.5))
    idx2 = np.where((gamma_s >= 1.5) & (gamma_s < 2.5))
    idx3 = np.where((gamma_s >= 2.5) & (gamma_s < 3.5))
    idx4 = np.where((gamma_s >= 3.5) & (gamma_s < 4.5))
    idx5 = np.where((gamma_s >= 4.5) & (gamma_s < 5))

    delta_ra = 1/(0.9*m + 9.4)
    delta_ra[idx0] = 0.0408
    delta_ra[idx1] = 0.0435
    delta_ra[idx2] = 0.0463
    delta_ra[idx3] = 0.0491
    delta_ra[idx4] = 0.0519
    delta_ra[idx5] = 0.0548

    series = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
    times = series.month.values
    TL_month = np.array([3.8, 4.2, 4.8, 5.2, 5.4, 6.4, 6.3, 6.1, 5.5, 4.3, 3.7, 3.6])
    T_L = np.zeros(365)
    p_div = np.exp(-z/8434.5)

    for i in range(12):
        T_L[np.where(times == 1+i)] = TL_month[0+i]

    EdirN0 = E0*np.exp(-T_L*m.T*delta_ra.T *p_div)
    EdirN0 = np.where(EdirN0 == E0, 0, EdirN0).T

    return EdirN0, gamma_s, theta_gen, p_div, T_L

def EGhor(N):
    EGhor0 = np.where(np.sin(gamma_s*dtor)==0, 0, (0.84*E0* np.sin(gamma_s.T*dtor)).T * np.exp(-0.027*p_div * T_L/np.sin(gamma_s*dtor).T).T)
    return EGhor0*(1- s*(N/8)**t)

def date_range(freq):
    date_range = pd.DataFrame({"date": pd.date_range(start="2022-01-01", end="2023-01-01", freq=freq)})
    return date_range["date"][:-1]

def Edirhor(N):
    Edirhor0 = EdirN0 * np.sin(gamma_s*dtor)
    if N==0:
        return Edirhor0
    else:
        return Edirhor0*(1 - N/8)

def Edirgen(N):
    Edirgen0 = EdirN0 * np.cos(theta_gen*dtor)
    E = np.where(Edirgen0 * (1- N/8) > 0,  Edirgen0 * (1- N/8), 0)
    return E

def Ediffhor(N):
    return EGhor(N) - Edirhor(N)

def Ediffgen(N):
    Ediffgen0 = np.where(gamma_s.T==0, 0, Ediffhor(0).T*(EdirN0.T/E0 * np.cos(theta_gen.T*dtor)/np.sin(gamma_s.T*dtor) + (1 - EdirN0.T/E0) * np.cos(gamma_E*dtor/2)**2)).T
    Ediffgen8 = Ediffhor(8)*np.cos(gamma_E/2)**2
    return Ediffgen0 * (1 - N/8) + Ediffgen8 * (N/8)

def plot_Ediffhor(N):
    plt.title("Yearly variation of $E_{diff,hor}$")
    plt.ylabel("E (W/m$^2$)")
    plt.plot(date_range("H"), Ediffhor(N).flatten(), label="Cloudy N=%i, yearly mean=%.2f W/m$^2$" %(N, np.mean(Ediffhor(N).flatten())))
    plt.plot(date_range("H"), Ediffhor(0).flatten(), label="Clear skies, yearly mean=%.2f W/m$^2$" %(np.mean(Ediffhor(0).flatten())))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_Ediffgen(N):
    plt.title("Yearly variation of $E_{diff,gen}$")
    plt.ylabel("E (W/m$^2$)")
    plt.plot(date_range("H"), Ediffgen(0).flatten(), label="Clear skies, yearly mean=%.2f W/m$^2$" %(np.mean(Ediffgen(0).flatten())))
    plt.plot(date_range("H"), Ediffgen(N).flatten(), label="Cloudy N=%i, yearly mean=%.2f W/m$^2$" %(N, np.mean(Ediffgen(N).flatten())))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_Edirhor(N):
    plt.title("Yearly variation of $E_{dir,hor}$")
    plt.ylabel("E (W/m$^2$)")
    #plt.plot(date_range("H"), Edirhor(0).flatten(), label="Clear skies, yearly mean=%.2f W/m$^2$" %(np.mean(Edirhor(0).flatten())))
    plt.plot(date_range("H"), Edirhor(N).flatten(), label="Cloudy N=%i, yearly mean=%.2f W/m$^2$" %(N, np.mean(Edirhor(N).flatten())))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    #plt.show()

def plot_Edirgen(N):
    plt.title("Yearly variation of $E_{dir,gen}$")
    plt.ylabel("E (W/m$^2$)")
    plt.plot(date_range("H"), Edirgen(0).flatten(), label="Clear skies, yearly mean=%.2f W/m$^2$" %(np.mean(Edirgen(0).flatten())))
    plt.plot(date_range("H"), Edirgen(N).flatten(), label="Cloudy N=%i, yearly mean=%.2f W/m$^2$" %(N, np.mean(Edirgen(N).flatten())))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_EGhor(N):
    plt.title("Yearly variation of $E_{G,hor}$")
    plt.ylabel("E (W/m$^2$)")
    #plt.plot(date_range("H"), EGhor(0).flatten(), label="Clear skies, yearly mean=%.2f W/m$^2$" %(np.mean(EGhor(0).flatten())))
    plt.plot(date_range("H"), EGhor(N).flatten(), label="Cloudy N=%i, yearly mean=%.2f W/m$^2$" %(N, np.mean(EGhor(N).flatten())))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    #plt.show()

EdirN0, gamma_s, theta_gen, p_div, T_L= pos(20,10)
plot_Edirhor(2)
EdirN0, gamma_s, theta_gen, p_div, T_L= pos()
plot_Edirhor(4.8)
EdirN0, gamma_s, theta_gen, p_div, T_L= pos(60,10)
plot_Edirhor(5.5)
plt.show()

EdirN0, gamma_s, theta_gen, p_div, T_L= pos()
plot_EGhor(5)
EdirN0, gamma_s, theta_gen, p_div, T_L= pos(60,10)
plot_EGhor(5)
plot_Edirgen(5)
plot_Edirhor(5)
plot_Ediffhor(5)
plot_Ediffgen(5)

def plot_delta():
    plt.plot(j, delta)
    plt.xlabel("days after 1. jan")
    plt.ylabel("$\delta (^\circ)$")
    plt.show()

def plot_ZGL():
    plt.plot(j, ZGL[:,0])
    plt.ylabel("$E_0$ $[W/m^2]$")
    plt.xlabel("days after 1. jan")
    plt.show()
