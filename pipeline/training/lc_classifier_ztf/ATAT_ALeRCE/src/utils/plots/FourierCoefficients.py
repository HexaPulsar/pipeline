import matplotlib.pyplot as plt
import seaborn


def get_fourier_coeffs(timefilm):
    for name,param in timefilm.named_parameters():
        if name in ['alpha_sin','alpha_cos','beta_sin', 'beta_cos']:
            fig = plt.figure(figsize=(16,5))
            weights = param.detach().cpu().numpy()
            ax1 =  fig.add_subplot(121)
            ax2 =  fig.add_subplot(122)
            sns.heatmap(weights.T,ax = ax1, cmap = 'bwr')
            sns.histplot(weights.T,ax =  ax2,bins = 20,color = 'bwr', edgecolor = 'w',alpha = 1)
            plt.legend(['h1','h2','h3','h4'])
            plt.suptitle(f"weights for param {name}")
            plt.show()