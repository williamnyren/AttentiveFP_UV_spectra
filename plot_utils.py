import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (ggplot, aes, geom_density, geom_vline, labs, theme_minimal,
                      theme, element_text, scale_x_continuous, geom_text)
import warnings

warnings.filterwarnings("ignore")

def make_density_plot(df, config, script_path):
    """Generate a density plot for MSE losses."""
    palette = ["blue", "green", "red", "purple", "orange"]
    quantiles = []
    quantiles_lst = [0.25, 0.5, 0.75, 0.95]
    for q in quantiles_lst: 
        quantiles.append(df['loss'].quantile(q))
    mean = df['loss'].mean()

    p = (ggplot(df, aes(x='loss')) +
         geom_density() +
         labs(y='Density', x='MSE loss', title='Density plot of MSE, Dataset: Test') +
         theme_minimal() +
         theme(legend_title=element_text(size=12, face='bold')) +
         scale_x_continuous(limits=[0, 0.05]))

    # Adding vertical lines and annotations
    for i, (quantile, label) in enumerate(zip(quantiles_lst, ['25%', '50%', '75%', '95%'])):
        p += geom_vline(xintercept=quantiles[i], color=palette[i+1], linetype="dashed")
        p += geom_text(x=quantiles[i], y=quantiles_lst[i]*50, label=label)

    # Add mean line and annotation
    p += geom_vline(xintercept=mean, color=palette[0], linetype="dashed")
    p += geom_text(x=mean, y=quantiles_lst[1]*50, label='Mean')


    # Save the plot
    p.save(os.path.join(script_path, 'plots', 'density_plot.png'), width=8, height=6, dpi=300, units='in')
    return 'density_plot.png'


def make_subplot(predictions, config):
    """Generate subplots for different prediction scenarios."""
    def plot_subplot(ax, data, title, x, x_text):
        """Helper function to plot subplots."""
        if not data.empty:
            y_pred, y_true, loss = data['y_pred'].iloc[0], data['y'].iloc[0], data['loss'].iloc[0]
            ax.plot(x, y_pred, label='Prediction')
            ax.plot(x, y_true, label='True', linestyle='--')
            #formatted_loss = f'{worst_prediction["loss"].values[0]:.4f}'
            #axs[0, 0].text(x_text_predicition, 0.85, f'Prediction: Worst', fontsize=12, ha='center')
            #axs[0, 0].text(x_text, 0.75, f'Loss: {formatted_loss}', fontsize=12, ha='center')
            #formated_smiles = worst_prediction['smiles'].values[0]
            #axs[0, 0].text(x_text_smiles, 0.5, f'smiles: {formated_smiles}', fontsize=12, ha='center')
            #axs[0, 0].set_title('Worst')
            ax.text(x_text, max(y_pred) * 0.95, f'Loss: {loss:.4f}', fontsize=12, ha='center')
            ax.set_title(title)
            ax.legend()

    wavelengths = np.linspace(config['min_wavelength'], config['max_wavelength'], config['out_dim'])
    x_text = wavelengths[int(0.75 * len(wavelengths))]

    fig, axs = plt.subplots(3, 2, figsize=(16, 20))  # Adjust size based on your needs
    scenarios = {
        'Worst': predictions.nlargest(1, 'loss'),
        'Best': predictions.nsmallest(1, 'loss'),
        'Average': predictions.iloc[(predictions['loss']-predictions['loss'].mean()).abs().argsort()[:1]],
        'Median': predictions.iloc[(predictions['loss']-predictions['loss'].median()).abs().argsort()[:1]],
        '25th percentile': predictions.iloc[(predictions['loss']-predictions['loss'].quantile(0.25)).abs().argsort()[:1]],
        '75th percentile': predictions.iloc[(predictions['loss']-predictions['loss'].quantile(0.75)).abs().argsort()[:1]]
    }

    for ax, (title, data) in zip(axs.flat, scenarios.items()):
        plot_subplot(ax, data, title, wavelengths, x_text)
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.2)

    plt.tick_params(axis='x', labelsize=12)  # Increase x-axis label size
    plt.tick_params(axis='y', labelsize=12)
   
    plt.draw()

    return fig

def make_lineplot(predictions):
    import seaborn as sns
    sns.lineplot(data=predictions, x='y_true', y='y_pred')