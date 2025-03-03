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
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Add some spacing between subplots
    plt.subplots_adjust(wspace=0.25)

    # Plot data in each subplot
    ax1.plot(df_train['step'],df_train['loss_train/not_weighted_var'],alpha = 0.3,color = 'blue')  # First subplot for variable 'a'
    ax1.set_title('Variance (1-var)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid('on')
    ax2.plot(df_train['step'],df_train['loss_train/not_weighted_inv'],alpha = 0.3,color = 'blue')  # Second subplot for variable 'b' 
    ax2.set_title('Invariance')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.grid('on')

    ax3.plot(df_train['step'],df_train['loss_train/not_weighted_cov'],alpha = 0.3,color = 'blue')  # Third subplot for variable 'c'
    ax3.set_title('Covariance') 
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss')
    ax3.grid('on')

    ax4.plot(df_train['step'],df_train['loss_train/loss'], alpha = 0.3,color = 'red')  # Third subplot for variable 'c'
    ax4.set_title('Total Loss') 
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Loss')
    ax4.grid('on')



    # Plot data in each subplot
    ax1.plot(df_val['step'],df_val['loss_validation/not_weighted_var'],color = 'blue')  # First subplot for variable 'a'
    ax1.set_title('Variance (1-var)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid('on')
    ax2.plot(df_val['step'],df_val['loss_validation/not_weighted_inv'],color = 'blue')  # Second subplot for variable 'b' 
    ax2.set_title('Invariance')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.grid('on')

    ax3.plot(df_val['step'],df_val['loss_validation/not_weighted_cov'],color = 'blue')  # Third subplot for variable 'c'
    ax3.set_title('Covariance') 
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss')
    ax3.grid('on')

    ax4.plot(df_val['step'],df_val['loss_validation/loss'], color = 'red')  # Third subplot for variable 'c'
    ax4.set_title('Total Loss') 
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Loss')
    ax4.grid('on')
    
    
    ax1.legend(['train', 'validation'])
    ax2.legend(['train', 'validation'])
    ax3.legend(['train', 'validation'])
    ax4.legend(['train', 'validation'])
    
    # Add a main title for the entire figure
    fig.suptitle('VICReg Loss', fontsize=16)
    # Display the plot
    plt.show()


    # Create figure and subplots
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Add some spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Plot data in each subplot
    ax1.plot(df_val['step'],df_val['loss_validation/not_weighted_var'])  # First subplot for variable 'a'
    ax1.set_title('Variance (1-var)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid('on')
    ax2.plot(df_val['step'],df_val['loss_validation/not_weighted_inv'])  # Second subplot for variable 'b' 
    ax2.set_title('Invariance')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.grid('on')

    ax3.plot(df_val['step'],df_val['loss_validation/not_weighted_cov'])  # Third subplot for variable 'c'
    ax3.set_title('Covariance') 
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss')
    ax3.grid('on')

    ax4.plot(df_val['step'],df_val['loss_validation/loss'], color = 'red')  # Third subplot for variable 'c'
    ax4.set_title('Total Loss') 
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Loss')
    ax4.grid('on')

    # Add a main title for the entire figure
    fig.suptitle('VICReg Loss (Validation)', fontsize=16)
    # Display the plot
    plt.show()