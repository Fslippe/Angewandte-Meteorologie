import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import matplotlib.dates as mdates
import scipy.stats as stats

plt.rcParams.update({'font.size': 22})

E0_C = 1368
dtor = 2*np.pi/360
j = np.linspace(1, 365, 365)
j_h = np.reshape(np.tile(j, (24)), (24, 365)).T
hours = np.linspace(1, 24, 24)
x = 0.98556*j - 2.72
xh =  0.98556*j_h - 2.72
lat_1443, lat_1684, lat_3032 = 48, 51.16, 53.81
lon_1443, lon_1684, lon_3032  = 7.85, 14.96, 10.71
z_1443, z_1684, z_3032 = 270, 238, 25

l_MEZ = 15
gamma_E = 30
alpha_E = 180
a = 0.506
b = 6.08
c = 1.6364

s = 0.72
t = 3.2
ex = 1 + 0.0334*np.cos(0.9856*dtor*j - 2.72/360*2*np.pi)
E0 = E0_C*ex
delta = np.arcsin(0.3978*np.sin((xh - 77.51 + 1.92*np.sin(xh*dtor))*dtor))/dtor
ZGL = -7.66*np.sin(xh*dtor) - 9.87 * np.sin((2*xh + 24.99 + 3.83*np.sin(xh*dtor))*dtor)

def pos(lon, lat, z):
    MOZ = hours - (l_MEZ - lon)*4/60
    WOZ = MOZ + ZGL/60
    omega = (WOZ - 12)*15
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

N_1443 = pd.read_csv("stundenwerte_N_01443_19490101_20211231_hist/produkt_n_stunde_19490101_20211231_01443.txt", sep=";")
N_1443["MESS_DATUM"] = pd.to_datetime(N_1443["MESS_DATUM"], format="%Y%m%d%H")
N_1443.index = N_1443["MESS_DATUM"]
N_1443.iloc[:,4] = np.where(N_1443.iloc[:,4] ==-1, np.nan, N_1443.iloc[:,4])
N_1443.iloc[:,4] = np.where(N_1443.iloc[:,4] > 8, 8, N_1443.iloc[:,4])
N_1443 = N_1443.asfreq("H", fill_value=np.nan)
mean_1443 = np.nanmean(N_1443.iloc[:,4].values)
N_1443["MESS_DATUM"] = N_1443.index

data = N_1443.iloc[:,4].resample("M").mean()
avg = data.groupby(data.index.month).mean()
avg.index = pd.to_datetime(avg.index, format="%m")
mask = np.isnan(N_1443.iloc[:,4].to_numpy())
x_1443 = N_1443.iloc[:,4].to_numpy()

for i in range(12):
    x_1443[mask] = np.where(N_1443.iloc[:,4].index.month.to_numpy()[mask] == avg.index.month.to_numpy()[i], avg.to_numpy()[i],  x_1443[mask])

N_1443.iloc[:,4] = x_1443

M_1443 = pd.read_csv("stundenwerte_ST_01443_row/produkt_st_stunde_19730101_20150131_01443.txt", sep=";")
M_1443["MESS_DATUM_WOZ"] = pd.to_datetime(M_1443["MESS_DATUM_WOZ"], format="%Y%m%d%H:%S")
M_1443.index = M_1443["MESS_DATUM_WOZ"]
M_1443["FG_LBERG"] = np.where(M_1443["FG_LBERG"] == -999, np.nan, M_1443["FG_LBERG"]/3600*1e4)
M_1443["FD_LBERG"] = np.where(M_1443["FD_LBERG"] == -999, np.nan, M_1443["FD_LBERG"]/3600*1e4)

N_1684 = pd.read_csv("stundenwerte_N_01684_19750701_20211231_hist/produkt_n_stunde_19750701_20211231_01684.txt", sep=";")
N_1684["MESS_DATUM"] = pd.to_datetime(N_1684["MESS_DATUM"], format="%Y%m%d%H")
N_1684.index = N_1684["MESS_DATUM"]
N_1684.iloc[:,4] = np.where(N_1684.iloc[:,4] ==-1, np.nan, N_1684.iloc[:,4])
N_1684.iloc[:,4] = np.where(N_1684.iloc[:,4] > 8, 8, N_1684.iloc[:,4])
N_1684 = N_1684.asfreq("H", fill_value=np.nan)
mean_1684 = np.nanmean(N_1684.iloc[:,4].values)
N_1684["MESS_DATUM"] = N_1684.index

data = N_1684.iloc[:,4].resample("M").mean()
avg = data.groupby(data.index.month).mean()
avg.index = pd.to_datetime(avg.index, format="%m")
mask = np.isnan(N_1684.iloc[:,4].to_numpy())
x_1684 = N_1684.iloc[:,4].to_numpy()

for i in range(12):
    x_1684[mask] = np.where(N_1684.iloc[:,4].index.month.to_numpy()[mask] == avg.index.month.to_numpy()[i], avg.to_numpy()[i],  x_1684[mask])

N_1684.iloc[:,4] = x_1684

M_1684 = pd.read_csv("stundenwerte_ST_01684_row/produkt_st_stunde_20010101_20220531_01684.txt", sep=";")
M_1684["MESS_DATUM_WOZ"] = pd.to_datetime(M_1684["MESS_DATUM_WOZ"], format="%Y%m%d%H:%S")
M_1684.index = M_1684["MESS_DATUM_WOZ"]
M_1684["FG_LBERG"] = np.where(M_1684["FG_LBERG"] == -999, np.nan, M_1684["FG_LBERG"]/3600*1e4)
M_1684["FD_LBERG"] = np.where(M_1684["FD_LBERG"] == -999, np.nan, M_1684["FD_LBERG"]/3600*1e4)

N_3032 = pd.read_csv("stundenwerte_N_03032_19490101_20211231_hist/produkt_n_stunde_19490101_20211231_03032.txt", sep=";")
N_3032["MESS_DATUM"] = pd.to_datetime(N_3032["MESS_DATUM"], format="%Y%m%d%H")
N_3032.index = N_3032["MESS_DATUM"]
N_3032.iloc[:,4] = np.where(N_3032.iloc[:,4] ==-1, np.nan, N_3032.iloc[:,4])
N_3032.iloc[:,4] = np.where(N_3032.iloc[:,4] > 8, 8, N_3032.iloc[:,4])
N_3032 = N_3032.asfreq("H", fill_value=np.nan)
mean_3032 = np.nanmean(N_3032.iloc[:,4].values)
N_3032["MESS_DATUM"] = N_3032.index

data = N_3032.iloc[:,4].resample("M").mean()
avg = data.groupby(data.index.month).mean()
avg.index = pd.to_datetime(avg.index, format="%m")
mask = np.isnan(N_3032.iloc[:,4].to_numpy())
x_3032 = N_3032.iloc[:,4].to_numpy()

for i in range(12):
    x_3032[mask] = np.where(N_3032.iloc[:,4].index.month.to_numpy()[mask] == avg.index.month.to_numpy()[i], avg.to_numpy()[i],  x_3032[mask])

N_3032.iloc[:,4] = x_3032
M_3032 = pd.read_csv("stundenwerte_ST_03032_row/produkt_st_stunde_19720101_20150131_03032.txt", sep=";")
M_3032["MESS_DATUM_WOZ"] = pd.to_datetime(M_3032["MESS_DATUM_WOZ"], format="%Y%m%d%H:%S")
M_3032.index = M_3032["MESS_DATUM_WOZ"]
M_3032["FG_LBERG"] = np.where(M_3032["FG_LBERG"] == -999, np.nan, M_3032["FG_LBERG"]/3600*1e4)
M_3032["FD_LBERG"] = np.where(M_3032["FD_LBERG"] == -999, np.nan, M_3032["FD_LBERG"]/3600*1e4)

def EGhor(N):
    EGhor0 = np.where(np.sin(gamma_s*dtor)==0, 0, (0.84*E0* np.sin(gamma_s.T*dtor)).T * np.exp(-0.027*p_div * T_L/np.sin(gamma_s*dtor).T).T)
    return EGhor0*(1- s*(N/8)**t)

def date_range(freq):
    date_range = pd.DataFrame({"date": pd.date_range(start="2022-01-01", end="2023-01-01", freq=freq)})
    return date_range["date"][:-1]

def Edirhor(N):
    Edirhor0 = EdirN0 * np.sin(gamma_s*dtor)
    Edirhor = np.where(N==0, Edirhor0, Edirhor0*(1 - N/8))
    return Edirhor

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
    plt.title("Yearly variation of $E_{dir,hor}$")
    plt.plot(date_range("H"), Ediffhor(N).flatten(), label="Cloudy")
    plt.plot(date_range("H"), Ediffhor(0).flatten(), label="Clear skies")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_Ediffgen(N):
    plt.title("Yearly variation of $E_{dir,hor}$")
    plt.plot(date_range("H"), Ediffgen(0).flatten(), label="Clear skies")
    plt.plot(date_range("H"), Ediffgen(N).flatten(), label="Cloudy")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_Edirhor(N):
    plt.title("Yearly variation of $E_{dir,hor}$")
    plt.plot(date_range("H"), Edirhor(0).flatten(), label="Clear skies")
    plt.plot(date_range("H"), Edirhor(N).flatten(), label="Cloudy")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_Edirgen(N, year):
    plt.title("Yearly variation of $E_{dir,gen}$")
    plt.plot(date_range("H"), Edirgen(N).flatten(), label="%i" %(year))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()

def plot_EGhor(N):
    plt.title("Yearly variation of $E_{dir,gen}$")
    plt.plot(date_range("H"), EGhor(0).flatten(), label="Clear skies")
    plt.plot(date_range("H"), EGhor(N).flatten(), label="Cloudy")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

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

idx2020 = pd.date_range('01-01-2020 00:00:00', '12-31-2020  23:00:00')
idx2021 = pd.date_range('01-01-2021', '12-31-2021')

def N_d(year=False, start="01-01-2000 00:00:00", end="01-31-2021 23:00:00", station=N_1443, mean=mean_1443):
    if year==False:
        mask = (station["MESS_DATUM"] >= start) & (station["MESS_DATUM"] <= end)
        N_d = station.loc[mask]
    else:
        N_d = station[station["MESS_DATUM"].dt.year == year]

    N_d.iloc[:,4] = np.where(N_d.iloc[:,4] > 8, 8, N_d.iloc[:,4])
    N_d.iloc[:,4] = np.where(N_d.iloc[:,4].values==-1, mean, N_d.iloc[:,4].values)

    return N_d

def plot_compare_Ediffgen(year1=2021, year2=2019):
    N_1 = N_d(year1, station=N_1443)
    N_2 = N_d(year2, station=N_1443)
    plt.plot(N_1["MESS_DATUM"], Ediffgen(N_1.iloc[:,4].values.reshape((365, 24))).flatten(), label="%i" %(year1))
    plt.plot(N_1["MESS_DATUM"], Ediffgen(N_2.iloc[:,4].values.reshape((365, 24))).flatten(), label="%i" %(year2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()
    plt.show()

def plot_compare_Edirhor(year1, year2=False, station=N_1443, station_nr=1443, mean=mean_1443):
    plt.title("Yearly variation of $E_{dir,hor}$ year %s station %s" %(year1, station_nr))
    if year2 == False:
        N_1 = N_d(year1, station=station, mean=mean)
        mean_1 = np.nanmean(Edirhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
        plt.plot(N_d(year1, station=station, mean=mean)["MESS_DATUM"][:8760], Edirhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten(), label="station %s year %s, yearly mean= %.2f W/m$^2$" %(station_nr, year1, mean_1))
    else:
        N_1 = N_d(year1, station=station)
        N_2 = N_d(year2, station=station)
        mean_1 = np.nanmean(Edirhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
        mean_2 = np.nanmean(Edirhor(N_2.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
        plt.plot(N_1["MESS_DATUM"][:8760], Edirhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten(), label="station %s year %s, yearly mean= %.2f W/m$^2$" %(station_nr, year1, mean_1))
        plt.plot(N_1["MESS_DATUM"][:8760], Edirhor(N_2.iloc[:,4].values[:8760].reshape((365, 24))).flatten(), label="station %s year %s, yearly mean= %.2f W/m$^2$" %(station_nr, year2, mean_2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()

def plot_compare_EGhor(year1, station=N_1443, station_nr=1443, mean=mean_1443):
    plt.title("Yearly variation of $E_{G,hor}$ year %s station %s" %(year1, station_nr))
    N_1 = N_d(year1, station=station, mean=mean)
    mean_1 = np.nanmean(EGhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
    plt.plot(N_d(year1, station=station, mean=mean)["MESS_DATUM"][:8760], EGhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten(), label="station %s, yearly mean= %.2f W/m$^2$" %(station_nr, mean_1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()

def plot_compare_Ediffhor(year1, station=N_1443, station_nr=1443, mean=mean_1443):
    plt.title("Yearly variation of $E_{diff,hor}$ station %s year %s" %(station_nr, year1))
    N_1 = N_d(year1, station=station, mean=mean)
    mean_1 = np.nanmean(Ediffhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
    plt.plot(N_d(year1, station=station, mean=mean)["MESS_DATUM"][:8760], Ediffhor(N_1.iloc[:,4].values[:8760].reshape((365, 24))).flatten(), label="VDI Model yearly mean= %.2f W/m$^2$" %( mean_1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.legend()


def mean_flux(mean_station, station):
    mean = np.zeros(35)
    for i in range(1986, 2021):
        N = N_d(i, station=station, mean=mean_station)
        mean[i-1986] = np.nanmean(EGhor(N.iloc[:,4].values[:8760].reshape((365, 24))).flatten())
    return mean

def M(station, year):
    return station[station["MESS_DATUM_WOZ"].dt.year == year]

def validate(station_nr1, station_nr2, year, nr):
    mean_G = np.nanmean(M(station_nr2, year)["FG_LBERG"])
    plt.title("Global radiation station %s year %s" %(station_nr1, year))
    plt.plot(M(station_nr2, year)["FG_LBERG"], label="Messured values yearly mean: %.2f W/m$^2$" %(mean_G))
    plot_compare_EGhor(year, station=station_nr1, station_nr=nr)
    plt.ylabel("W/m$^2$")
    plt.legend()
    plt.show()
    mean_D = np.nanmean(M(station_nr2, year)["FD_LBERG"])
    plt.title("Diffusive radiation station %s year %s" %(station_nr1, year))
    plt.plot(M(station_nr2, year)["FD_LBERG"], label="Messured values yearly mean: %.2f W/m$^2$" %(mean_D))
    plot_compare_Ediffhor(year, station=station_nr1, station_nr=nr)
    plt.ylabel("W/m$^2$")
    plt.legend()
    plt.show()
    mean_dir = mean_G - mean_D
    plt.plot(M(station_nr2, year)["FG_LBERG"] - M(station_nr2, year)["FD_LBERG"], label="Messured values yearly mean: %.2f W/m$^2$" %(mean_dir))
    plot_compare_Edirhor(year, station=station_nr1, station_nr=nr)
    plt.ylabel("W/m$^2$")
    plt.legend()
    plt.show()

EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1443, lat_1443, z_1443)

def validation():
    """VALIDATION - COMPARING WITH MESSURED DATA"""
    validate(N_1443, M_1443, 2014, 1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    validate(N_1684, M_1684, 2014, 1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    validate(N_3032, M_3032, 2014, 3032)
    validate(N_1443, M_1443, 2010, 1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    validate(N_1684, M_1684, 2010, 1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    validate(N_3032, M_3032, 2010, 3032)
    validate(N_1443, M_1443, 2011, 1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    validate(N_1684, M_1684, 2011, 1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    validate(N_3032, M_3032, 2011, 3032)
    validate(N_1443, M_1443, 2012, 1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    validate(N_1684, M_1684, 2012, 1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    validate(N_3032, M_3032, 2012, 3032)
    validate(N_1443, M_1443, 2013, 1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    validate(N_1684, M_1684, 2013, 1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    validate(N_3032, M_3032, 2013, 3032)



def regplot():
    """LINEAR REGRESSION VDI MODEL"""
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1443, lat_1443, z_1443)
    meanplot_1443 = mean_flux(mean_1443, N_1443)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    meanplot_1684 = mean_flux(mean_1684, N_1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    meanplot_3032 = mean_flux(mean_3032, N_3032)
    df = pd.DataFrame({"mean 1443": meanplot_1443, "mean 1684": meanplot_1684, "mean 3032": meanplot_3032})
    df.index = pd.date_range(start='1986', end='2021', freq="Y")
    df["mean 1443"] = np.where(df["mean 1443"].to_numpy() > 150, np.nan, df["mean 1443"].to_numpy())
    df.plot()
    plt.title("Yearly variations in average incoming direct solar radiation on horisontal surface")
    plt.ylabel("W/m$^2$")
    print("meanplot")
    plt.show()

    mask =  ~np.isnan(df["mean 1443"].to_numpy())
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(1986, 2021, len(df["mean 1443"].to_numpy()[mask])), df["mean 1443"].to_numpy()[mask])
    sns.regplot(np.linspace(1986, 2021, len(df["mean 1443"].to_numpy())), df["mean 1443"], label="Station 1443 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/df["mean 1443"].to_numpy()[0], r_value**2, pv))
    mask =  ~np.isnan(df["mean 1684"].to_numpy())
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(1986, 2021, len(df["mean 1684"].to_numpy()[mask])), df["mean 1684"].to_numpy()[mask])
    sns.regplot(np.linspace(1986, 2021, len(df["mean 1443"].to_numpy())), df["mean 1684"], label="Station 1684 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/df["mean 1684"].to_numpy()[0], r_value**2, pv))
    mask =  ~np.isnan(df["mean 3032"].to_numpy())
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(1986, 2021, len(df["mean 3032"].to_numpy()[mask])), df["mean 3032"].to_numpy()[mask])
    sns.regplot(np.linspace(1986, 2021, len(df["mean 1443"].to_numpy())), df["mean 3032"], label="Station 3032 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/df["mean 3032"].to_numpy()[0], r_value**2, pv))
    plt.legend()
    plt.title("Linear regression")
    plt.ylabel("W/m$^2$")
    plt.show()

def regplot_messured():
    """LINEAR REGRESSION MESSURED DATA"""
    m_1443 = M_1443["FG_LBERG"].resample("Y").mean()
    m_1443 = np.where(m_1443.to_numpy()<100, np.nan, m_1443.to_numpy())
    mask = ~np.isnan(m_1443)
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(1973, 2015, len(m_1443[mask])), m_1443[mask])
    sns.regplot(np.linspace(1973, 2015, len(m_1443)), m_1443, label="Station 1443 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/m_1443[mask][0], r_value**2, pv))

    m_1684 = M_1684["FG_LBERG"].resample("Y").mean()
    m_1684 = np.where(m_1684.to_numpy()<100, np.nan, m_1684.to_numpy())
    mask = ~np.isnan(m_1684)
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(2000, 2022, len(m_1684[mask])), m_1684[mask])
    sns.regplot(np.linspace(2000, 2022, len(m_1684)), m_1684, label="Station 1684 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/m_1684[mask][0], r_value**2, pv))

    m_3032 = M_3032["FG_LBERG"].resample("Y").mean()
    m_3032 = np.where(m_3032.to_numpy()<100, np.nan, m_3032.to_numpy())
    mask = ~np.isnan(m_3032)
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(1971, 2015, len(m_3032[mask])), m_3032[mask])
    sns.regplot(np.linspace(1971, 2015, len(m_3032)), m_3032, label="Station 3032 $y=$%.2f W/m$^2$ (%.2f%%) per year, $R^2=$%.2f, $p$-value:%.3f" %(slope, 100*slope/m_3032[mask][0], r_value**2, pv))
    plt.legend()
    plt.title("Linear regression")
    plt.ylabel("W/m$^2$")
    plt.show()

def compare():
    """COMPARISON DIFFERENT YEARS OR STATIONS"""
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1443, lat_1443, z_1443)
    plot_compare_Edirhor(2011)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    plot_compare_Edirhor(2011, station=N_3032, station_nr=3032, mean=mean_3032)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    plot_compare_Edirhor(2011, station=N_1684, station_nr=1684, mean=mean_1684)
    print("compare edirhor 2011")
    plt.show()
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_1684, lat_1684, z_1684)
    plot_compare_Edirhor(2019, station=N_1684, station_nr=1684, mean=mean_1684)
    EdirN0, gamma_s, theta_gen, p_div, T_L = pos(lon_3032, lat_3032, z_3032)
    plot_compare_Edirhor(2019, station=N_3032, station_nr=3032, mean=mean_3032)
    plt.show()

validation()
regplot()
regplot_messured()
compare()

#validation()
