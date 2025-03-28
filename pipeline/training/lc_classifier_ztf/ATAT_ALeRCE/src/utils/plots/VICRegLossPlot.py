import matplotlib.pyplot as plt
import numpy as np

def plot_four_loss(path:str):
    df  = pd.read_csv(path)
    df_train = df[['loss_train/loss','loss_train/not_weighted_cov',
       'loss_train/not_weighted_inv', 'loss_train/not_weighted_var',
       'loss_train/weighted_cov', 'loss_train/weighted_inv',
       'loss_train/weighted_var','step']].drop_duplicates()
    df_val = df[['loss_validation/loss', 'loss_validation/not_weighted_cov',
        'loss_validation/not_weighted_inv', 'loss_validation/not_weighted_var',
        'loss_validation/weighted_cov', 'loss_validation/weighted_inv',
        'loss_validation/weighted_var','step']].drop_duplicates()
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    # Create figure and subplots
    fig = plt.figure(figsize = (20,5))
    for i, train_col_name,val_col_name in zip(range(4),
                 ['loss_train/not_weighted_var',
                           'loss_train/not_weighted_inv',
                           'loss_train/not_weighted_cov',
                           'loss_train/loss'],
                 ['loss_validation/not_weighted_var',
                           'loss_validation/not_weighted_inv',
                           'loss_validation/not_weighted_cov',
                           'loss_validation/loss'],
                 ):
        ax = fig.add_subplot(int(141 + 1))
        ax.plot(df_train['step'],df_train[train_col_name],alpha = 0.3,color = 'blue')  # First subplot for variable 'a'
        ax.set_title('Variance (1-var)')
        ax.plot(df_val['step'],df_val[val_col_name],color = 'blue')  # First subplot for variable 'a'
        ax.set_title('Variance (1-var)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid('on')
        ax.legend(['train', 'validation'])
    
    
    # Add a main title for the entire figure
    fig.suptitle('VICReg Loss', fontsize=16)
    # Display the plot
    plt.show()
