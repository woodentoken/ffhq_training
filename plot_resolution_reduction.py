# plot the loss of the motion blur model via the loss.csv file

import matplotlib
import matplotlib.pyplot as plt

# create datafrane from the loss.csv file
import pandas as pd

train_data = pd.read_csv('training_loss.csv')

train_data['loss'] = train_data['loss']/train_data['loss'].min()

td_list = []
ax = plt.figure().add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Normalized Loss')
plt.rcParams.update({'font.size': 14})
resolutions = train_data['resolution'].unique()
for index, resolution in enumerate(resolutions):
    td = train_data[train_data['resolution'] == resolution]
    td.rename(columns={'loss': f'{resolution}x{resolution}'}, inplace=True)
    print(index)
    td.plot(
        ax=ax,
        x='epoch',
        y=f'{resolution}x{resolution}',
        xlabel='Epoch',
        kind='line',
        grid=True,
        title=f'Weighted Training Loss per Epoch vs. Image Resolution',
        fontsize=12,
        figsize=(8, 6),
        color=[1-(index/len(resolutions)), 0, 1*(index/len(resolutions)), 1],
        linewidth=2,
        style='o-'
    )
ax.margins(0.15)
ax.set_ylabel('Normalized Loss', fontdict={'fontsize': 14})
ax.set_xlabel('Epoch', fontdict={'fontsize': 14})
    #plt.plot(train_data['epoch'], train_data['loss'])
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig('resolution_training_loss.png')