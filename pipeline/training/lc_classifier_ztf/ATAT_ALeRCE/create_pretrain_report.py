import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import threading
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import time

import io
from datetime import datetime
import yaml
import torch
import torch.nn as nn 
from joblib import load
from tqdm import tqdm
import warnings
import umap

warnings.filterwarnings("ignore")
import glob
import os
from collections import OrderedDict
 
from src.layers.selfsupervised.lightcurve import LightCurveTransformer
from src.layers.selfsupervised.tabular import TabularTransformer
from src.data.modules.LitData import LitData
from src.layers.classifiers import TokenClassifier, MixedClassifier
from src.utils.umap_plots import *

def print_elapsed_time(start_time, stop_event):
    """Print elapsed time every second until the stop event is set."""
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        print(f"Time elapsed: {elapsed:.0f} seconds", end="\r")
        time.sleep(0.5)
        
class ReportPretraining:
    def __init__(
        self,
        path_to_training_dir: str,
        path_to_dataset: str,
        model_class,
        model_type: str,
        seed: 0,
        load_checkpoint:bool = True,
        device="cpu",
    ):
        self.class_dict = {
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
            "Microlensing": 21,
        }
        self.model_type = model_type
        self.checkpoint_src = path_to_training_dir
        self.yaml_args = self._load_yaml_args(path_to_training_dir)
        self.device = device
        self.load_ckpt = load_checkpoint
        self.dataloader = self._init_dataloader(
            path_to_dataset=path_to_dataset, set_type="test", seed=seed
        )
        self.model = self._init_model(model=model_class)
        self.preds_out, self.target = self._predict()

    def _load_yaml_args(self, path_args):
        path_args = glob.glob(f"{path_args}*args*")[0]
        with open(path_args, "r") as file:
            args = yaml.safe_load(file)
        return args

    def _init_dataloader(
        self, path_to_dataset, set_type="test", seed=0, batch_size=128
    ):
        self.yaml_args["general"]["data_root"] = path_to_dataset
        self.yaml_args["general"]["use_sampler"] = False
        self.yaml_args["general"]["batch_size"] = batch_size
        pl_datal = LitData(**self.yaml_args["general"])
        dataloader = pl_datal.test_dataloader()  # TODO set type implementation
        return dataloader

    def _init_model(self, model, checkpoint=None):
        model = model(**self.yaml_args[self.model_type])
        if self.load_checkpoint:
            checkpoint_path_clip = glob.glob(f"{self.checkpoint_src}*my_best_checkpoint*")
            print()
            print('using ckpt %s'.format(checkpoint_path_clip))
            print()
            checkpoint_clip = torch.load(
                checkpoint_path_clip[-1], map_location=torch.device(self.device)
            )
            od_atat = OrderedDict()
            for key in checkpoint_clip["state_dict"].keys():
                if "project" in key:
                    continue
                else:
                    od_atat[key.replace("model.transformer.", "")] = checkpoint_clip[
                        "state_dict"
                    ][key]
            model.load_state_dict(od_atat, strict=True)
        else:
            print('NO MODEL CKPT PROVIDED')
        return model

    def _predict(self):
        target = None
        preds_out = None
        self.model.eval().to(device=self.device)
        for b1 in tqdm(self.dataloader):
            b1 = {key: value.to(device=self.device) for key, value in b1.items()}
            t = b1["labels"]
            lc_emb = self.model(**b1)#[:, 0, :]

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
        return preds_out, target

    def _2d_umap_plot(
        self, n_neighbors, min_dist, metric, preds_out, target, title, type="by_class"
    ):
        print('creating UMAP reduction for data...')
        start_time = time.time()
        
        # Start timer thread
        stop_timer = threading.Event()
        timer_thread = threading.Thread(target=print_elapsed_time, args=(start_time, stop_timer))
        timer_thread.daemon = True
        timer_thread.start()
        
        # Run UMAP computation
        fig, axs = plt.subplots(1, 1, figsize=(25, 25))
        umap_model = umap.UMAP(
            n_components=2, metric=metric, min_dist=min_dist, n_neighbors=n_neighbors
        )
        umap_result = umap_model.fit_transform(preds_out)
        
        # Stop timer and print final time
        stop_timer.set()
        timer_thread.join()
        end_time = time.time()
        print(f'UMAP reduction completed in {end_time - start_time:.2f} seconds')
        
        plot_umap(
            axs, umap_result, target, 22, title
        ) if type == "by_class" else big_group_plot_umap(
            axs, umap_result, target, 22, title
        )
        return fig, axs

    def _3d_umap_plot(
        self, n_neighbors, min_dist, metric, preds_out, target, title, type="by_class"
    ):
        print('creating 3D UMAP reduction for data...')
        start_time = time.time()
        
        # Start timer thread
        stop_timer = threading.Event()
        timer_thread = threading.Thread(target=print_elapsed_time, args=(start_time, stop_timer))
        timer_thread.daemon = True
        timer_thread.start()
        
        # Run UMAP computation
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection="3d")
        umap_model = umap.UMAP(
            n_components=3, metric=metric, min_dist=min_dist, n_neighbors=n_neighbors
        )
        umap_result = umap_model.fit_transform(preds_out)
        
        # Stop timer and print final time
        stop_timer.set()
        timer_thread.join()
        end_time = time.time()
        print(f'3D UMAP reduction completed in {end_time - start_time:.2f} seconds')
        plot_umap_3d(ax, umap_result, target, 22, "UMAP 3D Visualization")
        return fig, ax




def create_classification_report_pdf(
    output_file="classification_report.pdf",
):
    test = ReportPretraining(
        path_to_training_dir="//home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/LC/DROPOUTwindow_v5/",
        path_to_dataset="data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12",
        model_class=LightCurveTransformer,
        model_type="lc",
        device="cuda:2",
        seed=0,
    )
    # Create a PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create a list to hold the PDF elements
    elements = []

    # Add title
    title_style = styles["Heading1"]
    elements.append(Paragraph("Classification Report", title_style))
    elements.append(Spacer(1, 0.25 * inch))

    # Add date
    date_style = styles["Normal"]
    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style
        )
    )
    elements.append(Spacer(1, 0.25 * inch))

    # Add classification report
    elements.append(Paragraph("Classification Metrics", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Get classification report as text
    report = "asdkjfa;ldjkf;lajd"

    # Format the report as a paragraph with monospace font
    report_style = ParagraphStyle(
        "ReportStyle", parent=styles["Normal"], fontName="Courier"
    )
    report_paragraph = Paragraph(f"<pre>{report}</pre>", report_style)
    elements.append(report_paragraph)
    elements.append(Spacer(1, 0.25 * inch))
    # Create the first confusion matrix
    elements.append(Paragraph("Confusion Matrix 1", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Generate confusion matrix and save as
    cm1_img_data = io.BytesIO()
    plt.figure(figsize=(8, 6))
    test._2d_umap_plot(15, 0.1, "euclidean", test.preds_out, test.target, "test")
    plt.tight_layout()
    plt.savefig(cm1_img_data, format="png")
    plt.close()

    # Add the image to the PDF
    cm1_img_data.seek(0)
    img1 = Image(cm1_img_data, width=6 * inch, height=4.5 * inch)
    elements.append(img1)
    elements.append(Spacer(1, 0.25 * inch))

    # Create the second confusion matrix if provided
    elements.append(Paragraph("Confusion Matrix 2", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Generate second confusion matrix and save as image
    cm2_img_data = io.BytesIO()
    plt.figure(figsize=(8, 6))

    test._3d_umap_plot(15, 0.1, "euclidean", test.preds_out, test.target, "test")

    plt.savefig(cm2_img_data, format="png")
    plt.close()

    # Add the image to the PDF
    cm2_img_data.seek(0)
    img2 = Image(cm2_img_data, width=6 * inch, height=4.5 * inch)
    elements.append(img2)

    # Build the PDF
    doc.build(elements)
    print(f"PDF report saved as {output_file}")


# Example usage
if __name__ == "__main__":
    # Generate the PDF report
    create_classification_report_pdf(
        output_file="iris_classification_report.pdf",
    )
