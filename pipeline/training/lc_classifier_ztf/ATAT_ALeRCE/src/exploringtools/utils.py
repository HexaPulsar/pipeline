import numpy as np





def get_confusion_matrix(preds,
                         target, 
                         taxonomy,
                         dataset_type:str, 
                         plot_title:str, 
                         order_classes:list[str]):
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        fs = 11
        y_true = [taxonomy.values_as_keys()[i] for i in np.array(target).astype(int)]
        y_pred = [taxonomy.values_as_keys()[i] for i in np.array(preds).astype(int)]

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=order_classes, normalize='true')
        np.set_printoptions(precision=4, suppress=True)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots(figsize=(11, 11)) #, dpi=110)
        decimals = 2
        im = ax.imshow(np.around(cm, decimals=decimals), interpolation='nearest', cmap=cmap)
        # color map
        new_color = cmap(1.0) 

        # Añadiendo manualmente las anotaciones con la media y desviación estándar
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    #print(cm[i, j])
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)
                else:
                    text = f'{np.around(cm[i, j], decimals=decimals)}'
                    color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)

        # Ajustes finales y mostrar la gráfica
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(np.arange(len(order_classes)))
        ax.set_yticks(np.arange(len(order_classes)))
        ax.set_xticklabels(order_classes)
        ax.set_yticklabels(order_classes)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        f1_ = classification_report(y_true,y_pred, target_names=list(taxonomy().keys()),digits = 4, output_dict=True)['macro avg']['f1-score']
        ax.set_title(f'{plot_title}: {dataset_type} | macro f1: {np.round(f1_,4)}', fontsize=16, pad=13)
        ax.set_xlabel('Predicted label', fontsize=16, labelpad=13)  # Label del eje x
        ax.set_ylabel('True label', fontsize=16, labelpad=13)        # Label del eje y

        #ax.xaxis.label.set_size(16)
        #ax.yaxis.label.set_size(16)
        #ax.xaxis.labelpad = 13
        #ax.yaxis.labelpad = 13
        return ax