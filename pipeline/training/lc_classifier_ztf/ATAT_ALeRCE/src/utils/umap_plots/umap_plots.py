
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm



def plot_umap(ax, umap_result, numeric_labels, num_classes, title):
    colors = ['#43aa8b', '#277da1', '#ca5cdd', '#277da1','#f9c74f',
              '#90be6d','#f8961e', '#f94144', '#f9844a',  '#ca5cdd', 
              '#f3722c','#277da1', '#43aa8b', '#577590', '#4d908e', 
              '#f9c74f','#90be6d', '#f94144', '#f3722c', '#f8961e', 
               '#277da1', '#ca5cdd']
    dict = {
    "AGN": 0,
    "QSO": 1,
    "EA": 2,
    "YSO": 3,
    "SNIa": 4,
    
    "CV/Nova": 5,
    "RRLc": 6,
    "RSCVn": 7,
    "Blazar": 8,
    "SNII": 9,
    
    "EB/EW": 10,
    "LPV": 11,
    "CEP": 12,
    "RRLab": 13,
    "Periodic-Other": 14,
    
    "DSCT": 15,
    "SNIbc": 16,
    "SLSN": 17,
    "TDE": 18,
    "SNIIb": 19,
    "SNIIn": 20,
    "Microlensing": 21
    }
    transient =  {
    "SNIa": 4,
    "SNII": 9,
    "SNIbc": 16,
    "SLSN": 17,
    "TDE": 18,
    "SNIIb": 19,
    "SNIIn": 20,
    "Microlensing": 21
    }
    for i in range(num_classes):
        class_indices = numeric_labels == i
        markers = ['o', 'o', 'D', 'o', '*',
                   'o', 'D', 'D', 'o', '*', 
                   'D', 'D', 'D', 'D', 'D',
                   'D', '*', '*', '*', '*',
                   '*', '*', '<', 'D', 'p']
        #if i in list(transient.values()):
        ax.scatter(x=umap_result[class_indices, 0], y=umap_result[class_indices, 1],s=10, marker=markers[i],label=list(dict.keys())[i], color=colors[i],alpha = 0.75 )
        #else:
        #    ax.scatter(x=umap_result[class_indices, 0], y=umap_result[class_indices, 1],s=10, marker=markers[i],label=list(dict.keys())[i], color='grey',alpha = 0.3 )
        

    ax.set_facecolor('black')
    
    ax.set_title(title)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    
    plt.legend()
    
    
def big_group_plot_umap(ax, umap_result, numeric_labels, num_classes, title):
    colors = ['cyan','yellow','magenta']
    transient =  {
    "SNIa": 4,
    "SNII": 9,
    "SNIbc": 16,
    "SLSN": 17,
    "TDE": 18,
    "SNIIb": 19,
    "SNIIn": 20,
    "Microlensing": 21
    }
    stochastic =  {
        "AGN": 0,
        "QSO": 1,
        "YSO": 3,
        "CV/Nova": 5,
        "Blazar": 8,
    }
    periodic =  {
        "EA": 2,
        "RRLc": 6,
        "RSCVn": 7,
        "EB/EW": 10,
        "LPV": 11,
        "CEP": 12,
        "RRLab": 13,
        "Periodic-Other": 14,
        "DSCT": 15,
    }
    
    for i in range(num_classes):
        class_indices = numeric_labels == i
        color = colors[0] if i in transient.values() else (colors[1] if i in stochastic.values() else colors[2])
        ax.scatter(x=umap_result[class_indices, 0], y=umap_result[class_indices, 1],s=7, color=color,alpha = 0.45) 
        

    ax.set_facecolor('black')
    
    ax.set_title(title)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.legend(['transient', 'stochastic','periodic'])
    

def plot_umap_3d(ax, umap_result, numeric_labels, num_classes, title):
    colors = ['#43aa8b', '#277da1', '#ca5cdd', '#277da1','#f9c74f',
              '#90be6d','#f8961e', '#f94144', '#f9844a',  '#ca5cdd', 
              '#f3722c','#277da1', '#43aa8b', '#577590', '#4d908e', 
              '#f9c74f','#90be6d', '#f94144', '#f3722c', '#f8961e', 
               '#277da1', '#ca5cdd']
    dict = {
        "AGN": 0, "QSO": 1, "EA": 2, "YSO": 3, "SNIa": 4,
        "CV/Nova": 5, "RRLc": 6, "RSCVn": 7, "Blazar": 8, "SNII": 9,
        "EB/EW": 10, "LPV": 11, "CEP": 12, "RRLab": 13, "Periodic-Other": 14,
        "DSCT": 15, "SNIbc": 16, "SLSN": 17, "TDE": 18, "SNIIb": 19,
        "SNIIn": 20, "Microlensing": 21
    }
    
    markers = ['o', 'o', 'D', 'o', '*',
                   'o', 'D', 'D', 'o', '*', 
                   'D', 'D', 'D', 'D', 'D',
                   'D', '*', '*', '*', '*',
                   '*', '*', '<', 'D', 'p'] # Repeat markers as needed
    
    for i in range(num_classes):
        class_indices = numeric_labels == i
        ax.scatter(umap_result[class_indices, 0], 
                  umap_result[class_indices, 1],
                  umap_result[class_indices, 2],
                  s=4, marker=markers[i],
                  label=list(dict.keys())[i], 
                  color=colors[i],
                  alpha=0.75)

    #ax.set_facecolor('black')
    ax.set_title(title)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    
    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Optional: set the viewing angle
    ax.view_init(elev=10, azim=45)




def plot_umap_3d_plotly(umap_result, numeric_labels, num_classes, title, output_path):
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Define the class groupings and colors
    transient = {
        "SNIa": 4, "SNII": 9, "SNIbc": 16, "SLSN": 17,
        "TDE": 18, "SNIIb": 19, "SNIIn": 20, "Microlensing": 21
    }
    stochastic = {
        "AGN": 0, "QSO": 1, "YSO": 3,
        "CV/Nova": 5, "Blazar": 8,
    }
    periodic = {
        "EA": 2, "RRLc": 6, "RSCVn": 7, "EB/EW": 10,
        "LPV": 11, "CEP": 12, "RRLab": 13,
        "Periodic-Other": 14, "DSCT": 15,
    }
    
    # Create class name mapping
    class_names = {
        0: "AGN", 1: "QSO", 2: "EA", 3: "YSO", 4: "SNIa",
        5: "CV/Nova", 6: "RRLc", 7: "RSCVn", 8: "Blazar", 9: "SNII",
        10: "EB/EW", 11: "LPV", 12: "CEP", 13: "RRLab", 14: "Periodic-Other",
        15: "DSCT", 16: "SNIbc", 17: "SLSN", 18: "TDE", 19: "SNIIb",
        20: "SNIIn", 21: "Microlensing"
    }
    
    # Create group labels
    group_labels = []
    for label in numeric_labels:
        if label in transient.values():
            group_labels.append('Transient')
        elif label in stochastic.values():
            group_labels.append('Stochastic')
        else:
            group_labels.append('Periodic')
    
    # Create class labels for hover text
    class_labels = [class_names[label] for label in numeric_labels]
    
    # Create the figure
    fig = go.Figure()
    
    # Color mapping
    color_map = {'Transient': 'green', 'Stochastic': 'red', 'Periodic': 'blue'}
    
    # Add traces for each group
    for group in ['Transient', 'Stochastic', 'Periodic']:
        mask = [label == group for label in group_labels]
        
        fig.add_trace(go.Scatter3d(
            x=umap_result[mask, 0],
            y=umap_result[mask, 1],
            z=umap_result[mask, 2],
            mode='markers',
            name=group,
            marker=dict(
                size=2,
                color=color_map[group],
                opacity=0.7
            ),
            text=[class_labels[i] for i in range(len(mask)) if mask[i]],  # Add class names as hover text
            hoverinfo='text'
        ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        width=1920,
        height=1080,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Show the plot in browser
    #fig.show()
    
    # Save the plot as HTML file
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


def plot_umap_3d_plotly_individual(umap_result, numeric_labels, num_classes, title, output_path):
    import plotly.graph_objects as go
    
    # Create class name mapping
    class_names = {
        0: "AGN", 1: "QSO", 2: "EA", 3: "YSO", 4: "SNIa",
        5: "CV/Nova", 6: "RRLc", 7: "RSCVn", 8: "Blazar", 9: "SNII",
        10: "EB/EW", 11: "LPV", 12: "CEP", 13: "RRLab", 14: "Periodic-Other",
        15: "DSCT", 16: "SNIbc", 17: "SLSN", 18: "TDE", 19: "SNIIb",
        20: "SNIIn", 21: "Microlensing"
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Generate a color palette for all classes
    colors = ['#43aa8b', '#277da1', '#ca5cdd', '#277da1','#f9c74f',
              '#90be6d','#f8961e', '#f94144', '#f9844a',  '#ca5cdd', 
              '#f3722c','#277da1', '#43aa8b', '#577590', '#4d908e', 
              '#f9c74f','#90be6d', '#f94144', '#f3722c', '#f8961e', 
              '#277da1', '#ca5cdd']
    
    # Add traces for each class
    for class_idx in range(num_classes):
        mask = numeric_labels == class_idx
        
        fig.add_trace(go.Scatter3d(
            x=umap_result[mask, 0],
            y=umap_result[mask, 1],
            z=umap_result[mask, 2],
            mode='markers',
            name=class_names[class_idx],
            marker=dict(
                size=2,
                color=colors[class_idx % len(colors)],
                opacity=0.98
            ),
            text=[class_names[class_idx]],
            hoverinfo='text'
        ))

    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        width=1920,
        height=1080,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
         
    )
    
    # Save the plot as HTML file
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")