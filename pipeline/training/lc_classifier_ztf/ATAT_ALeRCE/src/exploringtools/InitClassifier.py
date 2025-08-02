from .InitBackbone import InitBackbone
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


from .utils import get_confusion_matrix


ztf_order_classes = ['SNIa', # yes
                'SNIbc', # yes
                'SNIIb', # yes
                'SNII', # yes
                'SNIIn', # yes
                'SLSN', # yes
                'TDE', # yes
                'Microlensing', # yes
                'QSO',
                'AGN', # yes
                'Blazar',
                'YSO',
                'CV/Nova',
                'LPV',
                'EA',
                'EB/EW', # yes
                'Periodic-Other',
                'RSCVn',
                'CEP',
                'RRLab',
                'RRLc',
                'DSCT']

class InitClassifier(InitBackbone):
    def __init__(
        self,
        path_to_config_yaml,
        model,
        classifier,
        arg_key,
        use_lc = False,
        use_tab = False,
        use_mix = False,
        device = 'cpu'):
        super().__init__(path_to_config_yaml,
                        model,
                        arg_key,
                        device)
        self.classifier = classifier(lc_input_size = self.args.lc.embedding_size,
                                     inner_size = self.args.lc.embedding_size,
                 tab_input_size = self.args.tab.embedding_size,
                 use_lc = use_lc,
                 use_tab = use_tab,
                 use_mix = use_mix,
                 num_classes = self.args.num_classes,
                 dropout=0.0)

    def load_classifier_weights(self,weights: dict, strict = True):
        self.classifier.load_state_dict(weights, strict=strict)
        print("     - Loaded classifier weights")

    def predict(self,dataloader, device = None, pred_type = 'class'):
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
        self.backbone.eval().to(device=self.device)
        self.classifier.eval().to(device=self.device)

        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            emb = self.backbone(**b1)
            if pred_type == 'class':
                emb = self.classifier(emb)
                if isinstance(emb, dict):
                    if 'LC' in emb.keys():
                        output = emb["LC"]
                    if 'TAB' in emb.keys():
                        output = emb["TAB"]
                    if 'MIX' in emb.keys():
                        output = emb["MIX"]
            if pred_type == 'embeddings':
                output = emb
            preds_out = (
                np.concatenate([preds_out, output.detach().cpu().numpy()])
                if preds_out is not None
                else output.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.backbone.to(device="cpu")
        self.classifier.to(device="cpu")
        return preds_out, target

    def predict_by_time(self,dataloader, device = None, pred_type = 'class', eval_times = [8,16,32,64,128,256,512,1024,2048]):

        time_eval ={time:[] for time in eval_times}
        if device is not None:
            self.device = device
            print("device set to {}".format(device))
        target = None
        preds_out = None
        self.backbone.eval().to(device=self.device)
        self.classifier.eval().to(device=self.device)

        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            for time in eval_times:

                emb = self.backbone(**b1)
                if pred_type == 'class':
                    emb = self.classifier(emb)
                    if isinstance(emb, dict):
                        if 'LC' in emb.keys():
                            output = emb["LC"]
                        if 'TAB' in emb.keys():
                            output = emb["TAB"]
                        if 'MIX' in emb.keys():
                            output = emb["MIX"]
                if pred_type == 'embeddings':
                    output = emb
                preds_out = (
                    np.concatenate([preds_out, output.detach().cpu().numpy()])
                    if preds_out is not None
                    else output.cpu().detach().numpy()
                )
                time_eval[time] = preds_out
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.backbone.to(device="cpu")
        self.classifier.to(device="cpu")
        return preds_out, target



    def logistic_regression(self, X_train, y_train,X_test, y_test,taxonomy,knn_args:dict):
        std_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', LogisticRegression(random_state=0, max_iter = 500, multi_class = 'ovr'))
        ])

        # You can change n_neighbors as needed
        std_pipeline.fit(X_train,y_train)
        train_y_pred = std_pipeline.predict(X_train)

        classification = classification_report(y_train,train_y_pred, target_names=list(taxonomy.keys()),digits = 4)
        #print(classification)
        test_y_pred = std_pipeline.predict(X_test)

        classification = classification_report(y_test,test_y_pred, target_names=list(taxonomy.keys()),digits = 4)
        print(classification)

    def knn_classifier(self, X_train, y_train,X_test, y_test,taxonomy,knn_args:dict):

        knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('pca', PCA()),
        ('model', KNeighborsClassifier(**knn_args))
        ])

        knn_pipeline.fit(X_train, y_train)
        knn_preds = knn_pipeline.predict(X_test)

        classification = classification_report(y_test,knn_preds, target_names=list(taxonomy.keys()),digits = 4)
        print(classification)
        return knn_preds

    def get_confusion_matrix(self, preds,target, taxonomy, dataset_type:str, plot_title,  order_classes  ):
        return get_confusion_matrix(preds,target,taxonomy, dataset_type,plot_title, order_classes)
