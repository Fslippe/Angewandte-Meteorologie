import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import scipy.stats as stats
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
plt.rcParams.update({'font.size': 22})

rho = 1.225
h_mess = 10
h_hub = 101
cp = 0.45
A = 8012
U_min = 2
U_max = 25
n = 1/7
P_nenn = 3050e3

N_1443 = pd.read_csv("Wind_1443.csv")
N_1443["Zeitstempel | Stationen"] =  pd.to_datetime(N_1443["Zeitstempel | Stationen"], format="%Y%m%d")
N_1443.index = N_1443["Zeitstempel | Stationen"]
N_1443.iloc[:,1] = np.where(N_1443.iloc[:,1] == -999, np.nan, N_1443.iloc[:,1])

N_1684 = pd.read_csv("Wind_1684.csv")
N_1684["Zeitstempel | Stationen"] =  pd.to_datetime(N_1684["Zeitstempel | Stationen"], format="%Y%m%d")
N_1684.index = N_1684["Zeitstempel | Stationen"]
N_1684.iloc[:,1] = np.where(N_1684.iloc[:,1] == -999, np.nan, N_1684.iloc[:,1])

N_3032 = pd.read_csv("Wind_3032.csv")
N_3032["Zeitstempel | Stationen"] =  pd.to_datetime(N_3032["Zeitstempel | Stationen"], format="%Y%m%d")
N_3032.index = N_3032["Zeitstempel | Stationen"]
N_3032.iloc[:,1] = np.where(N_3032.iloc[:,1] == -999, np.nan, N_3032.iloc[:,1])

def potential(station, rho, A):
    P = station.iloc[:,1]**3*rho*A*0.5
    return P

def plot_potential(station, rho, A, label):
    potential(station, rho, A).plot(label="station %s" %(label))
    plt.xlabel("year")

    plt.legend()

def plot_wind_interval(station, nr):
    data = station.iloc[:,1]
    data.plot(linewidth=0.1, label = "Hourly data")
    data.resample("M").mean().plot(linewidth=2, label = "Monthly avereaged data")
    data.resample("Y").mean().plot(linewidth=5, label = "Yearly averaged data")

    mask =  ~np.isnan(data.resample("Y").mean().to_numpy())
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(0, 1, len(data.resample("Y").mean().to_numpy()[mask])), data.resample("Y").mean().to_numpy()[mask])
    plt.title("Wind data from station %s\n Trend in windspeed over time period: %.2f m/s, $P-value$: %.2f" %(nr, slope, pv))
    plt.xlabel("Date")
    plt.ylabel("Wind speed m/s")
    plt.legend()


def plot_potential_interval(station, nr):
    data = station.iloc[:,1]**3*rho*0.5
    data.plot(linewidth=0.1, label = "Hourly data")
    data.resample("M").mean().plot(linewidth=2, label = "Monthly avereaged data")
    data.resample("Y").mean().plot(linewidth=5, label = "Yearly averaged data")

    mask =  ~np.isnan(data.resample("Y").mean().to_numpy())
    print(data.resample("Y").mean().to_numpy()[mask])
    slope, intercept, r_value, pv, se = stats.linregress(np.linspace(0, 1, len(data.resample("Y").mean().to_numpy()[mask])), data.resample("Y").mean().to_numpy()[mask])
    print(slope)
    plt.title("Wind energy potential at station %s\n Trend in wind energy potential over time period: %.2f W/m$^2$, $P-value$: %.2f" %(nr, slope, pv))
    plt.xlabel("Date")
    plt.ylabel("Wind energy potential W/m$^2$")
    plt.legend()

def plot_wind_months(station, nr):
    plt.title("Average wind speeds in a year for station %s" %(nr))
    data = station.iloc[:,1].resample("D").mean()
    avg = data.groupby([data.index.month, data.index.day]).mean()
    avg.index = pd.date_range('01-01-2020', '31-12-2020')
    avg.plot(linewidth=0.5, label="dayly avereagd")
    resample = avg.resample("15D").mean()
    resample.plot(linewidth=2, label="15 day averaged")

    max = np.max(resample)
    argmax = np.argmax(resample)
    plt.plot(resample.index[argmax], max, "ro", markersize="10", label="15 day Maximum: %.2f m/s" %(max))
    min = np.min(resample)
    argmin = np.argmin(resample)
    plt.plot(resample.index[argmin], min, "bo", markersize="10", label="15 day Minimum: %.2f m/s" %(min))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.ylabel("Wind speed (m/s)")
    plt.legend()

def plot_potential_months(station, nr):
    data = station.iloc[:,1].resample("D").mean()
    avg = data.groupby([data.index.month, data.index.day]).mean()**3*rho*0.5
    avg.index = pd.date_range('01-01-2020', '31-12-2020')
    avg.plot(linewidth=0.5, label="dayly avereagd staation %s" %(nr))
    resample = avg.resample("15D").mean()
    resample.plot(linewidth=2, label="15 day averaged staation %s, Yearly mean: %.2f W/m$^2$" %(nr,avg.resample("Y").mean().to_numpy()[0] ))
    #plt.title("Average wind energy potential in a year for station %s, yearly mean %.2f W/m$^2$" %(nr, avg.resample("Y").mean()))
    plt.title("Average wind energy potential in a year for station %s" %(nr))

    max = np.max(resample)
    argmax = np.argmax(resample)
    plt.plot(resample.index[argmax], max, "ro", markersize="10", label="15 day Maximum: %.2f W/m$^2$" %(max))
    min = np.min(resample)
    argmin = np.argmin(resample)
    plt.plot(resample.index[argmin], min, "bo", markersize="10", label="15 day Minimum: %.2f W/m$^2$" %(min))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.ylabel("Wind energy potential $P/A$ (W/m$^2$)")
    plt.legend()

def plot_tech_potential(station, nr):
    data = station.iloc[:,1]
    data.iloc[:] = np.where((station.iloc[:,1].to_numpy()*(h_hub/h_mess)**n > 2) & (station.iloc[:,1].to_numpy()*(h_hub/h_mess)**n < 25), (station.iloc[:,1].to_numpy()*(h_hub/h_mess)**n)**3*cp*A*rho*0.5, 0)
    data.iloc[:] = np.where(data.to_numpy() < P_nenn, data.to_numpy(), P_nenn)*1e-3
    avg = data.groupby([data.index.month, data.index.day]).mean()
    avg.index = pd.date_range('01-01-2020', '31-12-2020')
    print(np.where(data.to_numpy()==0))
    print(len(np.where(data.to_numpy()==0)[0])/len(data.to_numpy())*100)
    print(len(np.where(data.to_numpy()==3050)[0])/len(data.to_numpy())*100)

    sns.histplot(data.to_numpy(), stat="percent", bins=50)
    plt.title("Histogram station %s" %(nr))
    plt.xlabel("Technical wind energy potential (KW)")
    plt.show()
    avg.plot(linewidth=0.5, label="dayly avereaged")
    resample = avg.resample("15D").mean()
    resample.plot(linewidth=2, label="15 day averaged")


    max = np.max(avg)
    argmax = np.argmax(avg)
    plt.plot(avg.index[argmax], max, "ro", markersize="10", label="Daily Maximum: %i KW" %(max))
    min = np.min(avg)
    argmin = np.argmin(avg)
    plt.plot(avg.index[argmin], min, "bo", markersize="10", label="Daily Minimum: %i KW" %(min))
    plt.title("Average technical wind energy potential (%i KW) in a year for station %s" %(avg.resample("Y").mean().to_numpy()[0], nr))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.ylabel("Technical wind energy potential $P_T$ (KW)")
    plt.legend()

plot_tech_potential(N_1443, 1443)
plt.show()
plot_tech_potential(N_1684, 1684)
plt.show()
plot_tech_potential(N_3032, 3032)
plt.show()

def potential_interval():
    plot_potential_interval(N_1443, 1443)
    plt.show()
    plot_potential_interval(N_1684, 1684)
    plt.show()
    plot_potential_interval(N_3032, 3032)
    plt.show()

def potential_months():
    plot_potential_months(N_3032, 3032)
    plot_potential_months(N_1684, 1684)
    plot_potential_months(N_1443, 1443)
    plt.show()

def potential():
    plt.title("Wind energy potential per m$^2$ $P/A$")
    plt.ylabel("W/m$^2$")
    plot_potential(N_3032, rho, 1, 3032)
    plot_potential(N_1443, rho, 1, 1443)
    plot_potential(N_1684, rho, 1, 1684)
    plt.show()

def wind_months():
    plot_wind_months(N_1684, 1684)
    plt.show()
    plot_wind_months(N_3032, 3032)
    plt.show()
    plot_wind_months(N_1443, 1443)
    plt.show()

def wind_interval():
    plot_wind_interval(N_1684, 1684)
    plt.show()
    plot_wind_interval(N_3032, 3032)
    plt.show()
    plot_wind_interval(N_1443, 1443)
    plt.show()
