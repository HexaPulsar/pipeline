
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from src.utils.plots.ATATConfusionMatrix import elasticc_confusion_matrix
from src.utils.data.AlerceDictionaries import ALERCE_TAXONOMY

def KNNClassifier(train:tuple,validation:tuple, test:tuple, plot_cm = False, cm_output_titple = ''):
    """_summary_

    Args:
        train (tuple): (train_pred, train_target)
        validation (tuple): (validation_pred,validation_target)
        test (tuple): (test_pred,test_target)
        plot_cm (bool, optional): whether to plot the confusion matrix for the results. Defaults to False.
        cm_output_titple (str, optional): _description_. Defaults to ''.
    """
    knn = KNeighborsClassifier(
        n_neighbors=10, weights="distance"
    )  # You can change n_neighbors as needed
    knn.fit(train[0],train[1])
        
    y_pred = knn.predict(validation[0])
    print(
        classification_report(
            validation[1], validation[0], target_names=list(ALERCE_TAXONOMY.all_classes.keys()), digits=4
        )
    )

    if plot_cm:
        out_metrics_balto = classification_report(
            target,
            y_pred,
            target_names=list(ALERCE_TAXONOMY.all_classes.keys()),
            output_dict=True,
        )["macro avg"]
        template_balto = ""
        for key in out_metrics_balto.keys():
            template_balto += " {} : {:.3f} ".format(
                key.upper(), out_metrics_balto[key]
            )
        fig, axes = plt.subplots(1, 1, figsize=(12, 12))
        elasticc_confusion_matrix(
            y_true=np.array(validation[1]).astype(int),
            y_pred=np.array(validation[0]).astype(int),
            classes=np.array(list(ALERCE_TAXONOMY.all_classes.keys())),
            ax=axes,
            normalize=True,
            title=f" FClassifier Results [TEST] \n\n {template_balto}",
        )
        plt.show()
    if test_dataloader is not None:
         
        y_pred = knn.predict(test[0])
        print(
            classification_report(
                test[1], test[0], target_names=list(ALERCE_TAXONOMY.all_classes.keys()), digits=4
            )
        )
        if plot_cm:
            out_metrics_balto = classification_report(
                target,
                y_pred,
                target_names=list(ALERCE_TAXONOMY.all_classes.keys()),
                output_dict=True,
            )["macro avg"]
            template_balto = ""
            for key in out_metrics_balto.keys():
                template_balto += " {} : {:.3f} ".format(
                    key.upper(), out_metrics_balto[key]
                )
            fig, axes = plt.subplots(1, 1, figsize=(12, 12))
            elasticc_confusion_matrix(
                y_true=np.array(test[1]).astype(int),
                y_pred=np.array(test[0]).astype(int),
                classes=np.array(list(ALERCE_TAXONOMY.all_classes.keys())),
                ax=axes,
                normalize=True,
                title=f" FClassifier Results [TEST] \n\n {template_balto}",
            )
            plt.show()