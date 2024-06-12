import pandas as pd
import os
import matplotlib.pyplot as plt

# load the data from the csv file
data = pd.read_csv('blur_loss.csv')
data.drop(columns=['Unnamed: 0'],inplace=True)

data['MAE loss'] = data['MAE loss']/data['MAE loss'].min()
data['MSE loss'] = data['MSE loss']/data['MSE loss'].min()
data['RMSE loss'] = data['RMSE loss']/data['RMSE loss'].min()

print(data)
data.plot(
    x='blur_degree',
    xlabel='Degree of Gaussian Motion Blur',
    ylabel='Normalized Loss',
    kind='line',
    grid=True,
    title='Blur Degree vs Loss, FFHQ 128x128',
    fontsize=12,
    figsize=(6, 4),
    linewidth=2,
    colormap='cool',
    style='.-'
    )
plt.tight_layout()
plt.savefig('blur_loss.png')  