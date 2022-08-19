import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
print('Modules are imported.')

# Permite mostrar más columnas cuando se usan comandos como .head()
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)

# Import COVID data set
corona_data_set_csv = pd.read_csv("covid19_Confirmed_dataset.csv")
print(corona_data_set_csv.to_string())

# Check the shape of the data frame (rows,columns)
print(corona_data_set_csv.shape)

# Delete useless columns
corona_data_set_csv.drop(["Lat","Long"],axis=1,inplace=True)  #Axis = 0 -> borrar filas
# Si no se pone inplace = True, el método creara otro data frame sin las columnas y deberá ser almacenado en otra variable.
print(corona_data_set_csv.to_string())

# We want data for countries, not region, so we will agregate province/state data according to their country.
# The method groupby creates a new data frame
corona_data_set_aggregated = corona_data_set_csv.groupby("Country/Region").sum()
print(corona_data_set_aggregated.to_string())
# The index became "Country/Region"
print(corona_data_set_aggregated.shape)

# Visualizing data for a country, example: China
corona_data_set_aggregated.loc["China"].plot()
# Compare it to "Italy"
corona_data_set_aggregated.loc["Italy"].plot()
# Add "Spain"
corona_data_set_aggregated.loc["Spain"].plot()
# Add "Chile"
corona_data_set_aggregated.loc["Chile"].plot()
plt.legend()
plt.show()


# Calculating a good measure
# Para este ejemplo, tomar los primeros 3 días
corona_data_set_aggregated.loc["China"][:3].plot()
plt.show()

# Calculate the maximum number of new cases in the period
corona_data_set_aggregated.loc["China"].diff().plot()
plt.show()

# We get the max value of the differentials
print(corona_data_set_aggregated.loc["China"].diff().max())
# Check Italy
print(corona_data_set_aggregated.loc["Italy"].diff().max())
# Check Spain
print(corona_data_set_aggregated.loc["Spain"].diff().max())

# Find the maximum infection rate of all countries
countries = list(corona_data_set_aggregated.index) #En nuestro ejemplo los paises corresponden a los indices
max_infection_rates = []

for c in countries:
    max_infection_rates.append(corona_data_set_aggregated.loc[c].diff().max())
corona_data_set_aggregated["max_infection_rate"] = max_infection_rates
print(corona_data_set_aggregated.head())

# Creamos un nuevo data frame para trabajar mas ordenado
corona_data = pd.DataFrame(corona_data_set_aggregated["max_infection_rate"])
print(corona_data.head())

# Importing happiness report
happiness_report_csv = pd.read_csv("worldwide_happiness_report.csv",encoding="UTF-8")
print(happiness_report_csv.head())

# Drop useless columns
useless_cols = ["Overall rank","Score","Generosity","Perceptions of corruption"]
happiness_report_csv.drop(useless_cols,axis=1,inplace=True)
print(happiness_report_csv.head())

# Change indices of data frame because will make the join process later easier
happiness_report_csv.set_index("Country or region",inplace=True)
print(happiness_report_csv.head())

# Now join the datasets
print(corona_data.shape)
print(happiness_report_csv.shape)

# The number of countries in corona_data is 187 and in happiness_report is 156

data  = corona_data.join(happiness_report_csv,how="inner")
print(data.to_string())

# Correlation matrix
print(data.corr())


# Visualization of results

# Plot GDP vs Max infection rate
x = data["GDP per capita"]
y = data["max_infection_rate"]
sns.scatterplot(x=x,y=y)
plt.show()
# To fix scaling visualization
sns.scatterplot(x=x,y=np.log(y))
plt.show()

# Another plot
sns.regplot(x=x,y=np.log(y))
plt.show()

# For other variables
x = data["Social support"]
y = data["max_infection_rate"]
sns.regplot(x=x,y=np.log(y))
plt.show()

x = data["Healthy life expectancy"]
y = data["max_infection_rate"]
sns.regplot(x=x,y=np.log(y))
plt.show()

x = data["Freedom to make life choices"]
y = data["max_infection_rate"]
sns.regplot(x=x,y=np.log(y))
plt.show()

# Bonus

confirmed_deaths = pd.read_csv("covid19_deaths_dataset.csv",encoding="UTF-8")

# Borrar columnas inútiles
confirmed_deaths.drop(["Lat","Long"],axis=1,inplace=True)
# Agrupar por países
confirmed_deaths_aggregated = confirmed_deaths.groupby("Country/Region").sum()
print(confirmed_deaths_aggregated.to_string())

# Encontrar el total de muertes para cada país
countries = list(confirmed_deaths_aggregated.index) #En nuestro ejemplo los paises corresponden a los indices
total_deaths = []

for country in countries:
    total_deaths.append(confirmed_deaths_aggregated.loc[country].sum())
confirmed_deaths_aggregated["total deaths"] = total_deaths
print(confirmed_deaths_aggregated.to_string())

# Creamos data frame para análisis
bonus_data = pd.DataFrame(confirmed_deaths_aggregated["total deaths"])
death_data = bonus_data.join(happiness_report_csv,how="inner")
print(death_data.to_string())
print(death_data.corr())

x = death_data["GDP per capita"]
y = death_data["total deaths"]
plt.bar(x,y,edgecolor="black",width=0.5)
plt.xlabel("GDP")
plt.ylabel("Total Deaths")
plt.show()



# El gráfico de barras que sale acá es interesante, parece ser que se muere más gente en países de GDP medio.
# Los bajos podrían no ser reportados.

# Otra mirada en scatter con escala logarítmica

import warnings
warnings.filterwarnings("ignore")     #Para ignorar las advertensias de logaritmos de 0

plt.scatter(x,np.log(y))
plt.xlabel("GDP")
plt.ylabel("Log(Total Deaths)")
plt.show()