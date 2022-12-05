#
# stats.py
# Author: Ahmad Abdalmageed
# Date: 7/23/21
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load Saved Models Logs
history_heart = pd.concat([pd.read_csv('./Stats/Unet_5x2_32F_logs.csv'), pd.read_csv('./Stats/Unet_5x2_32F_logs3.csv')])
print(history_heart.tail())
# Loss and Dice
train_dice_loss = pd.DataFrame(list(zip(history_heart['dice_coef'], 1 + history_heart['loss'])),
                               columns=['dice', 'loss'])
valid_dice_loss = pd.DataFrame(list(zip(history_heart['val_dice_coef'], 1 + history_heart['val_loss'])),
                               columns=['dice', 'loss'])
sns.set_theme(style="darkgrid")

# Plot Training and Loss
fig1, ax1 = plt.subplots(figsize=(20, 10))
sns.lineplot(data=train_dice_loss, ax=ax1, palette='flare', linewidth=3, dashes=False)
ax1.legend(['Dice Score', 'Loss'])
plt.title('Heart Segmentation Training', size=20)
plt.xlabel('Epochs', size=15)
plt.ylabel('Training Dice Score', size=15)
plt.savefig('Heart_Train.png')

fig2, ax2 = plt.subplots(figsize=(20, 10))
sns.lineplot(data=valid_dice_loss, ax=ax2, palette='flare', linewidth=3, dashes=False)
ax2.legend(['Dice Score', 'Loss'])
plt.title('Heart Segmentation Validation', size=20)
plt.xlabel('Epochs', size=15)
plt.ylabel('Validation Dice Score', size=15)
plt.savefig('Heart_valid.png')

