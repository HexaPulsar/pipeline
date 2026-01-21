from src.augmentations import LightCurveTransform as LC
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Optional, Literal
import pytorch_lightning as pl
from torch.optim.lr_scheduler import  SequentialLR,ConstantLR,CosineAnnealingWarmRestarts,CosineAnnealingLR, LinearLR, ExponentialLR
import logging
from sklearn.metrics.pairwise import cosine_similarity
from lion_pytorch import Lion
from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY,ZTF_TAXONOMY
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

class PretrainModule(pl.LightningModule):
    def __init__(self,model,loss,lr = 0.001,eval_knn = False,eval_regressor = True, **kwargs):
        """
        Batch must be a tuple of batches: (batch, augmented batch)
        Model output must be of shape (bsz,embeddings)
        Loss input is (embedding_batch_1, embedding_batch_2)
        Args:
            model (_type_): _description_
            loss (_type_): _description_

        """
        super().__init__()
        self.gradients_ = None
        self.lr = lr
        self.model = model
        self.loss = loss
        logging.debug('using learning rate {}'.format(self.lr))
        self.init_model()
        self.collect_train_embs = None
        self.collect_train_labels = None
        self.collect_val_embs = None
        self.collect_val_labels = None
        self.eval_knn = eval_knn
        self.eval_regressor = eval_regressor

    def init_model(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def training_step(self, batch, batch_idx):
        embedding_1 =self.model(**batch[0])[:,0,:] #torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        embedding_2 = self.model(**batch[1])[  :,0,:]#torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)

        #embedding_1  =torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        #embedding_2 = torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)
        loss_dict = self.loss(embedding_1,
                              embedding_2)

        with torch.no_grad():
            for key,value in loss_dict.items():
                if 'emb_corr' in key:
                    self.logger.experiment.add_histogram(key, value,self.global_step)
                elif 'mean_' in key:
                    self.logger.experiment.add_histogram(key, value,self.global_step)

                elif 'percent' in key:
                    self.log(f'{key}', value ,on_epoch=False,on_step=True, sync_dist=True)
                else:
                    self.log(f'loss_train/{key}', value ,on_epoch=False,on_step=True, sync_dist=True)

        #self.log(f'Tmax_0',self.model.time_encoder.time_encoders[0].Tmax,on_step = True, sync_dist=True)
        return loss_dict['loss']
    '''
    def on_validation_epoch_start(self, dataloader_idx):
        for dataloader_idx in range(3):
            if dataloader_idx == 0:
                return
            else:
                self.collect_train_embs = None
                self.collect_train_labels = None
                self.collect_val_embs = None
                self.collect_val_labels = None
        return super().on_validation_batch_start()
    '''
    def validation_step(self, batch, batch_idx):
        embedding_1 = self.model(**batch[0])[:,0,:]#torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        embedding_2 =self.model(**batch[1])[:,0,:] #torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)


        #embedding_1  =torch.concat([embedding[:,0,:] for embedding in self.model(**batch[0]).values()], dim = -1)
        #embedding_2 = torch.concat([embedding[:,0,:] for embedding in self.model(**batch[1]).values()], dim = -1)
        loss_dict = self.loss(embedding_1,
                              embedding_2)
        with torch.no_grad():
            for key,value in loss_dict.items():
                    if 'emb_corr' not in key:
                        self.log(f'loss_validation/{key}', value ,on_epoch=True,on_step=False, add_dataloader_idx=False)
        return loss_dict['loss']
        '''x
        elif dataloader_idx == 1:
            labels = batch.pop('labels')
            embs = self.model(**batch)

            self.collect_train_embs = (
                np.concatenate([self.collect_train_embs, embs.detach().cpu().numpy()])
                if self.collect_train_embs is not None
                else embs.cpu().detach().numpy()
            )
            self.collect_train_labels = (
                np.concatenate([self.collect_train_labels, labels.detach().cpu().numpy()])
                if self.collect_train_labels is not None
                else labels.cpu().detach().numpy()
            )
            return 0
        elif dataloader_idx ==2:
            labels = batch.pop('labels')
            embs = self.model(**batch)
            self.collect_val_embs = (
                np.concatenate([self.collect_val_embs, embs.detach().cpu().numpy()])
                if self.collect_val_embs is not None
                else embs.cpu().detach().numpy()
            )
            self.collect_val_labels = (
                np.concatenate([self.collect_val_labels, labels.detach().cpu().numpy()])
                if self.collect_val_labels is not None
                else labels.cpu().detach().numpy()
            )
            return 0
        '''
    '''
    def on_validation_epoch_end(self, dataloader_idx):

        for dataloader_idx in range(3):
            if dataloader_idx == 2:
                taxonomy = ZTF_TAXONOMY()
                class_names = list(taxonomy.keys())
                labels = list(taxonomy.values())  # assuming label encoding is 0 to 21

                if self.eval_regressor:
                    lr_report = classification_report(
                        self.collect_val_labels,
                        self.get_regressor_eval(),
                        labels=labels,
                        target_names=class_names,
                        digits=4,
                        output_dict=True
                    )
                    for key,value in dict(lr_report['macro avg']).items():
                        if key =='support':
                            continue
                        self.log(f'LRegressor/{key}', value ,on_epoch=True,on_step=False, add_dataloader_idx=False, sync_dist=True)
                    self.log(f'LRegressor/accuracy', np.round(lr_report['accuracy'],4) ,on_epoch=True,on_step=False, add_dataloader_idx=False, sync_dist=True)
                    del lr_report
                if self.eval_knn:
                    knn_report = classification_report(
                        self.collect_val_labels,
                        self.get_knn_eval(),
                        labels=labels,
                        target_names=class_names,
                        digits=4,
                        output_dict=True
                    )
                    for key,value in dict(knn_report[ 'macro avg']).items():
                        if key =='support':
                            continue
                        self.log(f'KNN_3/{key}', value ,on_epoch=True,on_step=False, add_dataloader_idx=False, sync_dist=True)
                    self.log(f'KNN_3/accuracy', np.round(knn_report['accuracy'],4) ,on_epoch=True,on_step=False, add_dataloader_idx=False, sync_dist=True)
                    del knn_report

                del taxonomy, class_names, labels

               # to_print = print(classification_report(self.collect_val_labels,val_preds, target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=False))
               # self.logger.experiment.add_text(to_print, self.current_epoch)
               # self.logger.experiment.add_scalars(main_tag="PerClassF1",
               #                                    tag_scalar_dict=scalars,
               #                                    global_step = self.current_epoch,
               #                                    sync_dist = True)

        return super().on_validation_epoch_end()
    '''
    def test_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        self.warmup = 1000
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        constant = ConstantLR(optimizer,1)
        #cosine = CosineAnnealingWarmRestarts(optimizer,T_0=100,eta_min=1e-6)
        linear = LinearLR(optimizer, start_factor=1e-2, total_iters=self.warmup)
        cosine = CosineAnnealingWarmRestarts(optimizer,T_0=self.warmup,eta_min=1e-8)

        scheduler = SequentialLR(
                    optimizer,
                    schedulers=[constant,constant],
                    milestones=[self.warmup]
                )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def get_real_classes_weights(self,labels):

        class_sample_count = np.array(
            [
                len(np.where(labels == t)[0])
                for t in np.unique(labels)
            ]
        )
       # print('class_sampler_count', class_sample_count)
        weight = 1.0 / class_sample_count
        uniques = np.unique(labels).astype(int)
        d = {key: value for key, value in zip(uniques, weight)}
        samples_weight = np.array([d[labels[i].item()] for i in range(len(labels))])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight


    def get_regressor_eval(self,):
        weights = self.get_real_classes_weights(torch.tensor(self.collect_train_labels))
        weights_dict = {float(i):  weights[i] for i in range(len(weights))}

        std_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        ('model', LogisticRegression(random_state=0, max_iter = 1000, multi_class = 'ovr', class_weight=weights_dict))
        ])
        std_pipeline.fit(self.collect_train_embs,self.collect_train_labels)
        val_preds = std_pipeline.predict(self.collect_val_embs)
        return val_preds

    def get_knn_eval(self):
        knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z = (x - mean) / std
        #('pca', PCA()),
        ('model', KNeighborsClassifier(n_neighbors=3,weights = 'distance'))
        ])
        knn_pipeline.fit(self.collect_train_embs,self.collect_train_labels)
        val_preds = knn_pipeline.predict(self.collect_val_embs)
        return val_preds