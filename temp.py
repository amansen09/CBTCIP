import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment data
data = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Data preprocessing
# ...

# Data visualization
plt.plot(data['Year'], data['UnemploymentRate'])
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate Over Time')
plt.show()
