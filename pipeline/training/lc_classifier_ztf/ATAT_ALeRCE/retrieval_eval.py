import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.modules.LitPretrain import LitPretrain
from src.layers.transformer.ATAT import LightCurveTransformer
from src.layers.utils.projector import VICRegProjector
from src.losses.VICReg import VICReg
from src.models.PretrainModule import PretrainModule

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_pretrained_model(checkpoint_path, cfg):
    """Load pretrained SSL model for LC only."""
    transformer = LightCurveTransformer(**cfg.lc)
    projector = VICRegProjector(
        VICReg(
            cfg.vicreg.inv_coeff,
            cfg.vicreg.var_coeff,
            cfg.vicreg.cov_coeff,
            cfg.datamodule.batch_size
        ),
        cfg.vicreg.shape_projector_1,
        cfg.vicreg.shape_projector_2
    )
    pl_model = PretrainModule(
        model=transformer,
        loss=projector,
        lr=cfg.learning_rate,
        eval_regressor=False,
        **cfg
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    pl_model.load_state_dict(state_dict)

    return pl_model.eval()


def generate_embeddings(model, dataloader, device):
    """Generate embeddings for all samples in dataloader."""
    embeddings = []
    labels = []

    model = model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            # Handle SSL batch format: (batch_dict, labels) or ((data_dict, aug_dict), labels)
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data, batch_labels = batch

                # If batch_data is itself a tuple (SSL augmented pair), take first
                if isinstance(batch_data, (tuple, list)):
                    batch_data = batch_data[0]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # Move batch to device
            if isinstance(batch_data, dict):
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch_data.items()}
            else:
                batch_data = batch_data.to(device)

            # Get embeddings (CLS token)
            output = model.model(**batch_data)
            emb = output[:, 0, :].cpu()
            embeddings.append(emb)
            labels.append(batch_labels.cpu() if isinstance(batch_labels, torch.Tensor) else torch.tensor(batch_labels))

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings.numpy(), labels.numpy()


def compute_retrieval_metrics(embeddings, labels, k_values=[1, 3, 5, 10]):
    """Compute retrieval metrics: recall@k and precision@k."""
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = StandardScaler().fit_transform(embeddings)
    sim_matrix = cosine_similarity(embeddings)

    metrics = {}

    for k in k_values:
        if k >= len(embeddings):
            continue

        recalls = []
        precisions = []

        for i in range(len(embeddings)):
            neighbors_idx = np.argsort(-sim_matrix[i])[1:k+1]
            neighbor_labels = labels[neighbors_idx]
            query_label = labels[i]

            correct = (neighbor_labels == query_label).sum()
            recall = 1.0 if correct > 0 else 0.0
            recalls.append(recall)

            precision = correct / k
            precisions.append(precision)

        metrics[f'recall@{k}'] = np.mean(recalls)
        metrics[f'precision@{k}'] = np.mean(precisions)

    return metrics


def compute_knn_accuracy(embeddings, labels, k=3):
    """Compute k-NN classification accuracy."""
    from sklearn.neighbors import KNeighborsClassifier

    embeddings = StandardScaler().fit_transform(embeddings)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)

    predictions = knn.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)

    return accuracy
