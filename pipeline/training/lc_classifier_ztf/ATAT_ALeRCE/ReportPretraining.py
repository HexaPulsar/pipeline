import numpy as np
import pandas as pd
from tqdm import tqdm 
import glob
from copy import deepcopy

class Report:
    def __init__(
        self,
        path_to_training_dir, 
        path_to_dataset, 
        model_class, model_type, 
        custom_parse_key_str,
        taxonomy, 
        seed, 
        device, 
        batch_size, 
        load_checkpoint, 
        **kwargs,
    ):
        
        self.cfg = self._load_yaml_args(path_to_training_dir).ATATConfig
        self.taxonomy = taxonomy
        self.path_to_dataset = path_to_dataset 
        self.device = device
        self.load_checkpoint = load_checkpoint 
        self.seed = seed
        self.custom_parse_key_str = custom_parse_key_str
        self.model_type = model_type
        self.checkpoint_src = path_to_training_dir
        self.batch_size = batch_size
        
    @staticmethod
    def _load_yaml_args(path_args):
        import yaml
        from  hydra.utils import instantiate
        path_args = glob.glob(f"{path_args}/.hydra/*config*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        args =  instantiate(args)
        return args

    def _init_dataloader(
        self, path_to_dataset, set_type="test", seed=0,  
    ):
        from src.data.modules.LitData import LitData
        from collections import OrderedDict
        dataset_exclude_transform_keys ={
            key: value 
            for key, value in self.cfg.datamodule.dataset.items() 
            if key not in ['transforms_1','transforms_2', 'data_root', 'batch_size']
        }
        modded_config = deepcopy(self.cfg)
        modded_config.datamodule.dataset = dataset_exclude_transform_keys
        modded_config.datamodule.dataset.data_root = self.path_to_dataset
        modded_config.datamodule.batch_size = self.batch_size
        
        pl_datal = LitData(**modded_config.datamodule)
        if set_type == "train":
            dataloader = pl_datal.train_dataloader()
        elif set_type == "validation":
            dataloader = pl_datal.val_dataloader()
        elif set_type == "test":
            dataloader = pl_datal.test_dataloader()
        return dataloader
    
    def _predict(self,dataloader):
        target = None
        preds_out = None
        self.model.eval().to(device=self.device)
        for b1 in tqdm(dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            lc_emb = self.model(**b1)  # [:, 0, :]
            preds_out = (
                np.concatenate([preds_out, lc_emb.detach().cpu().numpy()])
                if preds_out is not None
                else lc_emb.cpu().detach().numpy()
            )
            target = (
                np.concatenate([target, t.cpu().detach().numpy()])
                if target is not None
                else t.detach().cpu().numpy()
            )
        self.model.to(device="cpu")
        return preds_out, target

class ReportClassification(Report):
    def __init__(self, 
                 path_to_training_dir, 
                 path_to_dataset, 
                 taxonomy,
                 model_class, 
                 model_type, 
                 custom_parse_key_str,
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True, 
                 **kwargs):
        
        super().__init__(path_to_training_dir, 
                         path_to_dataset, 
                         model_class, model_type, 
                         custom_parse_key_str,
                         taxonomy, 
                         seed, 
                         device, 
                         batch_size, 
                         load_checkpoint, 
                         **kwargs)
        self.model = self._init_model(model_class)
        self.taxonomy = taxonomy

    def _init_model(self,model ):
        from src.layers.ClassifierBaseModel import ClassifierBaseModel
        from src.layers.classifiers import TokenClassifier
        from torch import device, load
        from collections import OrderedDict
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        classifier = TokenClassifier(num_classes=self.cfg.num_classes,**self.cfg.lc )
        model = ClassifierBaseModel(model, classifier)
        if self.checkpoint_src is not None or self.load_checkpoint:
            
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*classifier_ckpt*")
            print(checkpoint_path_clip)
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if 'projection' in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def classification_report(self, dataset_type: str = 'validation', digits = 4, confusion_matrix = True):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_out, target = self._predict(dataloader)
        preds_out = np.argmax(preds_out, axis = -1)
        classification = classification_report(target,preds_out, target_names=list(self.taxonomy().keys()),digits = digits)
        print(classification)
        if confusion_matrix:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 1, figsize=(12, 12))
            axes = self.get_confusion_matrix(preds_out,target, title = 'Classifier Results [{}]'.format(dataset_type.upper()))
        return axes

    def get_confusion_matrix(self, preds,target, title = ''):
        from src.utils.plots.ATATConfusionMatrix import elasticc_confusion_matrix
        from sklearn.metrics import classification_report
        out_metrics_balto = classification_report(
            target, preds,target_names=list(self.taxonomy().keys()), output_dict=True
        )["macro avg"]
        template_balto = ""
        for key in out_metrics_balto.keys():
            template_balto += " {} : {:.3f} ".format(key.upper(), out_metrics_balto[key])
        
        return elasticc_confusion_matrix(
            y_true=np.array(target).astype(int),
            y_pred=np.array(preds).astype(int),
            classes= np.array(list(self.taxonomy().keys())),
            ax=None,
            normalize=True,
            title=title,
        )

class ReportPretraining(Report):
    
    def __init__(
        self, 
        path_to_training_dir, 
        path_to_dataset, 
        taxonomy,
        model_class, 
        model_type, 
        custom_parse_key_str,
        seed = 0, 
        device="cpu", 
        batch_size=128, 
        load_checkpoint = True,
        figsize = (20,20),
        umap_args={"n_neighbors": 15, "min_dist": 0.25, "metric": "euclidean"},
        marker_size = 4
    ):
        super().__init__(path_to_training_dir=path_to_training_dir,
                        path_to_dataset=path_to_dataset,
                        model_type=model_type,
                        model_class=model_class,
                        taxonomy=taxonomy,
                        custom_parse_key_str=custom_parse_key_str,
                        seed=seed,
                        device= device,
                        batch_size=batch_size,
                        load_checkpoint=load_checkpoint)
        self.model = self._init_model(model_class)
        self.taxonomy = taxonomy
        self.figsize =figsize
        self.marker_size = marker_size
        self.umap_args = umap_args 

    def mapper(self, set_type = 'validation'):
        import pandas as pd
        import umap.plot
        from bokeh.resources import INLINE
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        umap.plot.output_notebook(resources=INLINE)
        preds_out, target = self._predict(dataloader)
        hover_data = pd.DataFrame({'index':range(len(preds_out)),
                                'label':target})
        hover_data['item'] = hover_data.label.map({value:key for key,value in self.taxonomy().items()})
        preds_out = (preds_out/preds_out.max())*1000
        mapper = umap.UMAP(**self.umap_args).fit(preds_out)
        return hover_data,target, mapper
    
    def generate_umap_plots(self, set_type = 'validation'):
        import matplotlib.pyplot as plt
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=set_type, seed=self.seed
        )
        self.preds_out, self.target = self._predict(dataloader)
        #umap_result = self._umap(n_components=2, **self.umap_args)
        #self._2d_umap_plot(umap_result, self.target, "by_class", type_="by_class")
        #plt.show() 
        #self._2d_umap_plot(umap_result, self.target, "by_hierarchy", type_="by_hierarchy")
        #plt.show()
        umap_result = self._umap(n_components=3, **self.umap_args)
        self._3d_umap_plot(umap_result, self.target, "3d", type_="by_class")
        plt.show()
        

    def _init_model(self,model ):
        from torch import device, load
        from collections import OrderedDict
        model = model(**self.cfg.lc) if self.model_type =='lc' else model(**self.cfg.tab)
        if self.checkpoint_src is not None or self.load_checkpoint:
            print(glob.glob(self.checkpoint_src))
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*pretrain_ckpt*")
            print('using checkpoint {}'.format(checkpoint_path_clip[-1].split('=')[-1]))
            checkpoint_clip = load(
                checkpoint_path_clip[-1], map_location=device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if 'model' not in key:
                    continue
                od_atat[key.replace(f"{self.custom_parse_key_str}", "")] = checkpoint_clip[
                    "state_dict"
                ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO CKPT LOADED')
        return model
    
    def _umap(self, n_components, metric, min_dist, n_neighbors):
        import umap
        print(f"creating {n_components}D UMAP reduction for data...")
        umap_model = umap.UMAP(
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        )
        umap_result = umap_model.fit_transform(self.preds_out)
        return umap_result

    def _2d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap,big_group_plot_umap
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size=self.marker_size
        ) if type_ == "by_class" else big_group_plot_umap(
            axs, umap_result, target, len(self.taxonomy), title,marker_size =self.marker_size
        )
        plt.savefig('{}/2d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, axs

    def _3d_umap_plot(self, umap_result, target, title, type_="by_class"):
        import matplotlib.pyplot as plt
        from src.utils.plots.umap_plots import plot_umap_3d
        fig = plt.figure(figsize=self.figsize, facecolor='black')
        ax = fig.add_subplot(111, projection="3d")
        plot_umap_3d(ax, umap_result, target, len(self.taxonomy), "UMAP 3D Visualization",marker_size=self.marker_size)
        plt.savefig('{}/3d_umap_{}.pdf'.format(self.checkpoint_src,type_))
        return fig, ax

    def _knn(self, plot_cm=True):
        from src.utils.KNNClassifier import KNNClassifier
        train_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="train", seed=self.seed
        )
        val_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="validation", seed=self.seed
        )
        test_dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type="test", seed=self.seed
        )
        KNNClassifier(self._predict(train_dataloader),
                      self._predict(val_dataloader),
                      self._predict(test_dataloader), plot_cm=True)
        
class ComparePerformance:
    def __init__(self, 
                 path_1,
                 path_2,
                 path_to_dataset, 
                 model_class, 
                 model_type,  
                 custom_parse_key_str, 
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True ):
        self.taxonomy = None
        self.model_1 = ReportClassification(path_1, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
        self.model_2 = ReportClassification(path_2, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
    
    def compare(self, dataset_type:str= 'validation', eval_time = 2048):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_1, target = self.model_1._predict(dataloader)
        preds_2,_ = self.model_2._predict(dataloader)
        cr_1  = classification_report(np.argmax(preds_1, axis = -1), target, output_dict=True)
        cr_2  = classification_report(np.argmax(preds_2, axis = -1), target, output_dict=True)

        f1_1 = []
        f1_2 = []

        fig,ax = plt.subplots(1,1,figsize = (8,5))
        for key,value in cr_1.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_1.append(value['f1-score'])
        df_1 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_1 })
        ordered_df_1 = df_1#.sort_values(by='values')
        my_range_1=range(1,len(df_1.index)+1)
        

        for key,value in cr_2.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_2.append(value['f1-score'])
        df_2 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_2})
        ordered_df_2 = df_2#.sort_values(by='values')
        my_range_2=range(1,len(df_2.index)+1)
        plt.plot(ordered_df_1['values'].values, my_range_1, "o", alpha = 0.8,color='blue')
        plt.plot(ordered_df_2['values'].values, my_range_2, "o", alpha = 0.8,color='red')

        plt.legend(['1','2'])
        
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_1, xmin=0, xmax=ordered_df_1['values'], color='blue', alpha = 0.5)
        
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_2, xmin=0, xmax=ordered_df_2['values'], color='red', alpha = 0.5)

        # Add titles and axis names
        plt.yticks(my_range_1, ordered_df_1['group'])
        plt.title(f"F1-Score LC Classifier Comparison for Eval Time {eval_time} ", loc='center')

        plt.xlabel('F1-Score')
        plt.ylabel('Class')
        plt.xlim(0,1)
        plt.grid('on', )
        # Show the plot
        plt.show()