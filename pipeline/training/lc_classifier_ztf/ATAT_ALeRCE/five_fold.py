from src.layers.transformer.ATAT import LightCurveTransformer, TabularTransformer

from src.utils.data.AlerceDictionaries import ZTF_TAXONOMY
import numpy as np
import pandas as pd

from src.exploringtools.InitClassifier import InitClassifier, ztf_order_classes
from src.exploringtools.InitDataloader import InitDataLoader
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier

from sklearn.metrics import classification_report

import re
def summarize_classification_reports(reports):
    """
    Given a list of sklearn classification_report dictionaries (with output_dict=True),
    compute the mean and standard deviation across them and return a formatted string report.
    """
    # Convert each report to a DataFrame and collect them
    df_list = [pd.DataFrame(r).transpose() for r in reports]
    # print(df_list)
    # Align all DataFrames, using outer join to include all possible labels
    all_labels = sorted(set().union(*[df.index for df in df_list]))
    df_list = [df.reindex(all_labels) for df in df_list]

    # Combine and group by label
    combined = pd.concat(df_list).groupby(level=0)

    # Compute mean and std
    mean_df = combined.mean()
    std_df = combined.std()

    # Format the output string
    report_str = ""
    metrics = ["precision", "recall", "f1-score"]

    for label in mean_df.index:
        if label == "accuracy":
            # Accuracy is a single float, stored in the "precision" column
            acc_mean = mean_df.loc[label].get("precision", float("nan"))
            acc_std = std_df.loc[label].get("precision", float("nan"))
            report_str += (
                f"{label:>12} & & &  {acc_mean:.4f} $\\pm$ {acc_std:.4f} & \\\ \n"
            )
        else:
            values = []
            for metric in metrics:
                mean_val = mean_df.loc[label].get(metric, float("nan"))
                std_val = std_df.loc[label].get(metric, float("nan"))
                values.append(f"{mean_val:.4f} $\\pm$ {std_val:.4f}")
            support = mean_df.loc[label].get("support", float("nan"))
            report_str += (
                f"{label:>12} & " + " & ".join(values) + f"  & {int(support):>5} \\\ \n"
            )

    return report_str


def classification_report_to_latex(report_str):
    """
    Converts a classification report string into LaTeX tabular format,
    ensuring proper formatting for Overleaf (e.g., spacing after \\\\).
    """
    # Clean up and extract lines
    lines = report_str.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    # Identify the lines with data
    data_lines = [line for line in lines if re.match(r"^[\w\- ]+\s+\d+\.\d+", line)]

    # Begin LaTeX tabular
    top = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & Precision & Recall & F1-score & Support \\ ",
        r"\midrule",
    ]
    latex = []
    top = "\n".join(top)
    print(top)
    for line in data_lines:
        # if 'accuracy' in line:
        #   print(line)
        parts = re.split(r"\s{2,}", line.strip())
        # print(parts)
        if len(parts) == 5:  # class line with support
            class_name, precision, recall, f1, support = parts
            latex.append(f"{class_name} & {precision} & {recall} & {f1} & {support}\\")
        elif len(parts) == 4:  # summary line (accuracy, macro avg, etc.)
            class_name, precision, recall, f1 = parts
            latex.append(r"\midrule")
            latex.append(f"{class_name} & {precision} & {recall} & {f1}\\")
        elif len(parts) == 3:  # summary line (accuracy, macro avg, etc.)
            class_name, accuracy, support = parts
            latex.append(r"\midrule")
            latex.append(f"{class_name} & & & {accuracy} & {support}\\")

    for string in latex:
        print(string + "\\")
    bottom = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Classification Report}",
        r"\label{tab:classification_report}",
        r"\end{table}",
    ]
    bottom = "\n".join(bottom)
    print(bottom)



def local_five_fold(type_ ='lc', use_lc = True, use_tab = True, use_mix= True):

    for EXPERIMENT_NAME in EXPERIMENT_NAME_LIST:
        for SUBSET in ["test"]:
            TRANSIENT = []
            STOCHASTIC = []
            PERIODIC = []
            for i in range(5):
                # SETUP MODEL
                print("Using seed:{}".format(i))
                PATH = f"{FIVE_FOLD_DIRECTORY}/{EXPERIMENT_NAME}_{i}/"
                test = InitClassifier(
                    path_to_config_yaml=PATH,
                    model=LightCurveTransformer if type_ =='lc' else TabularTransformer,
                    classifier=MultimodalClassifier,
                    use_lc=use_lc,
                    use_tab = use_tab,
                    use_mix = use_mix,
                    arg_key=type_,
                )
                backbone_od = test.create_ordered_dict(
                    remove_if_in_key_list=[
                        "projection",
                        "transformer_tab" if type == 'tab' else 'transformer_lc',
                        "classifier",
                    ],
                    rename_keys=("model.", ""),
                    checkpoint_name="classifier_ckpt",
                )
                classifier_od = test.create_ordered_dict(
                    remove_if_in_key_list=[
                        "projection",
                        "model",
                    ],
                    rename_keys=("classifier.", ""),
                    checkpoint_name="classifier_ckpt",
                )
                test.load_backbone_weights(backbone_od)
                test.load_classifier_weights(classifier_od)
                test.args.datamodule.val_use_sampler = False

                # INIT DATALOADER
                dl = InitDataLoader(
                    update_dataset_path=DATASET,
                    update_batch_size=16,
                    datamodule_args=test.args.datamodule,
                )
                (
                    dl.init_test_dataset()
                    if SUBSET == "test"
                    else dl.init_validation_dataset()
                )

                # INFERENCE
                val_preds, val_true = (
                    test.predict(dl.test_dataset, DEVICE)
                    if SUBSET == "test"
                    else test.predict(dl.validation_dataset, DEVICE)
                )
                # REPORT RESULTS
                import pandas as pd

                df = pd.DataFrame(
                    {"true": val_true, "pred": np.argmax(val_preds, axis=-1)}
                )

                transient_dict = ZTF_TAXONOMY.transient.group
                transients = df.query(
                    "true in {}".format(list(transient_dict.values()))
                )

                ##
                stochastic_dict = ZTF_TAXONOMY.stochastic.group
                stochastics = df.query(
                    "true in {}".format(list(stochastic_dict.values()))
                )
                ####
                periodic_dict = ZTF_TAXONOMY.periodic.group
                periodics = df.query("true in {}".format(list(periodic_dict.values())))


                TRANSIENT.append(classification_report(
                        transients["true"],
                        transients["pred"],
                        target_names=list(ZTF_TAXONOMY.transient.group.keys()),
                        labels=list(ZTF_TAXONOMY.transient.group.values()),
                        digits=4,
                        output_dict=True,
                    ))

                STOCHASTIC.append(classification_report(
                        stochastics["true"],
                        stochastics["pred"],
                        target_names=list(ZTF_TAXONOMY.stochastic.group.keys()),
                        labels=list(ZTF_TAXONOMY.stochastic.group.values()),
                        digits=4,
                        output_dict=True,
                    ))


                PERIODIC.append(classification_report(
                        periodics["true"],
                        periodics["pred"],
                        target_names=list(ZTF_TAXONOMY.periodic.group.keys()),
                        labels=list(ZTF_TAXONOMY.periodic.group.values()),
                        digits=4,
                        output_dict=True,
                    ))

            with open(
                f"BY_SUPERCLASS_{SUBSET}_{EXPERIMENT_NAME.split('/')[-1]}_classification_report.txt",
                "w",
            ) as f:
                f.write(summarize_classification_reports(TRANSIENT))
                f.write("\n")
                f.write(30*"===")
                f.write("\n")
                f.write(summarize_classification_reports(STOCHASTIC))
                f.write("\n")
                f.write(30*"===")
                f.write("\n")
                f.write(summarize_classification_reports(PERIODIC))
            print(30 * "==")


def global_five_fold(type_, use_lc, use_tab, use_mix):

    for EXPERIMENT_NAME in EXPERIMENT_NAME_LIST:
        for SUBSET in ["test"]:
            REPORT = []
            for i in range(5):
                # SETUP MODEL
                print("Using seed:{}".format(i))
                PATH = f"{FIVE_FOLD_DIRECTORY}/{EXPERIMENT_NAME}_{i}/"
                print(PATH)
                test = InitClassifier(
                    path_to_config_yaml=PATH,
                    model=LightCurveTransformer if type_ =='lc' else TabularTransformer,
                    classifier=MultimodalClassifier,
                    use_lc=use_lc,
                    use_tab = use_tab,
                    use_mix = use_mix,
                    arg_key=type_,
                )
                backbone_od = test.create_ordered_dict(
                    remove_if_in_key_list=[
                        "projection",
                       "transformer_tab" if type == 'tab' else 'transformer_lc',
                        "classifier",
                    ],
                    rename_keys=("model.", ""),
                    checkpoint_name="classifier_ckpt",
                )
                classifier_od = test.create_ordered_dict(
                    remove_if_in_key_list=[
                        "projection",
                        "model",
                    ],
                    rename_keys=("classifier.", ""),
                    checkpoint_name="classifier_ckpt",
                )
                test.load_backbone_weights(backbone_od)
                test.load_classifier_weights(classifier_od)
                test.args.datamodule.val_use_sampler = False

                # INIT DATALOADER
                dl = InitDataLoader(
                    update_dataset_path=DATASET,
                    update_batch_size=16,
                    datamodule_args=test.args.datamodule,
                )
                (
                    dl.init_test_dataset()
                    if SUBSET == "test"
                    else dl.init_validation_dataset()
                )

                # INFERENCE
                val_preds, val_true = (
                    test.predict(dl.test_dataset, DEVICE)
                    if SUBSET == "test"
                    else test.predict(dl.validation_dataset, DEVICE)
                )
                # REPORT RESULTS
                report = classification_report(
                    val_true,
                    np.argmax(val_preds, axis=-1),
                    target_names=list(ZTF_TAXONOMY().keys()),
                    digits=4,
                    output_dict=True,
                )
                with open(f"{PATH}/{SUBSET}_classification_report.txt", "w") as f:
                    f.write(
                        classification_report(
                            val_true,
                            np.argmax(val_preds, axis=-1),
                            target_names=list(ZTF_TAXONOMY().keys()),
                            digits=4,
                        )
                    )
                REPORT.append(report)
                # PLOT CM
                ax = test.get_confusion_matrix(
                    np.argmax(val_preds, axis=-1),
                    val_true,
                    dataset_type=SUBSET,
                    taxonomy=ZTF_TAXONOMY,
                    plot_title=PLOT_TITLE,
                    order_classes=ztf_order_classes,
                )
                # Save the figure
                ax.figure.savefig(
                    f"{PATH}/CM_{SUBSET}_{PLOT_TITLE.replace(' ',"_")}_{i}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            print(summarize_classification_reports(REPORT))
            with open(
                f"{SUBSET}_{EXPERIMENT_NAME.split('/')[-1]}_classification_report.txt",
                "w",
            ) as f:
                f.write(summarize_classification_reports(REPORT))
            print(30 * "==")


# DATASET = "/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dataset.h5"
DATASET = "/home/mdelafuente/ZTF_SSL_Dataset/data/H5_files/BY_PARTITION/200_FF.h5"
DEVICE = "cuda:3"
FIVE_FOLD_DIRECTORY = "/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/"
EXPERIMENT_NAME_LIST = [
     "ABLATION/LC/class_BASELINE_TF",
     "ABLATION/LC/class_CONV_TF",
     "ABLATION/LC/class_TF_GELU_NORM",
     "ABLATION/LC/class_TF_GELU_NORM_EXP",
     "ABLATION/LC/class_TF_GELU_NORM_EXP_VEL",
     "ABLATION/LC/class_TF_GELU_NORM_EXP_VEL_ACC",
     "ABLATION/LC/class_TF_GELU_NORM_EXP_VEL_ACC_SEQNORM",
    # "AUGMENTATIONS/LC/class_band_permute",
    # "AUGMENTATIONS/LC/class_windows",
    # "AUGMENTATIONS/LC/class_time_gauss_factor",
    # "AUGMENTATIONS/LC/class_roll",
    # "AUGMENTATIONS/LC/class_data_factor",
    # "AUGMENTATIONS/LC/class_random_factor",
     #"AUGMENTATIONS/LC/class_random_factor_05",
     #"AUGMENTATIONS/LC/class_all_p1",
    # "AUGMENTATIONS/LC/class_all_p1_128"
    #"AUGMENTATIONS/LC/class_datatime_gauss_factor",
]

EXPERIMENT_NAME_LIST = [#"SCALING/LC/class_1_32",
                        #"SCALING/LC/class_1_64",
                        #"SCALING/LC/class_1_128",

                        #"SCALING/LC/class_2_32",
                        #"SCALING/LC/class_2_64",
                        #"SCALING/LC/class_2_128",

                        #"SCALING/LC/class_3_32",
                        #"SCALING/LC/class_3_64",
                        #"SCALING/LC/class_3_128",




                        #"SCALING/LC/v2_TAB_class_1_32",
                       # "SCALING/LC/v2_TAB_class_1_64",
                       # "SCALING/LC/v2_TAB_class_1_128",

                       # "SCALING/LC/v2_TAB_class_2_32",
                      #  "SCALING/LC/v2_TAB_class_2_64",
                      #  "SCALING/LC/v2_TAB_class_2_128",

                       # "SCALING/LC/v2_TAB_class_3_32",
                       # "SCALING/LC/v2_TAB_class_3_64",
                       # "SCALING/LC/v2_TAB_class_3_128",
                        'SCALING/LC_MD_FEAT/MOSPERF_class_'


]
#PLOT_TITLE = "ZTF-ATAT LC ONLY"
#global_five_fold('tab', use_lc=False, use_tab=True, use_mix = False)
#local_five_fold('tab', use_lc=False, use_tab=True, use_mix = False)



#PLOT_TITLE = "ZTF-ATAT TAB ONLY"
#global_five_fold('tab', use_lc=False, use_tab=True, use_mix = False)
#local_five_fold('tab', use_lc=False, use_tab=True, use_mix = False)


PLOT_TITLE = "ZTF-ATAT MULTIMODAL"
global_five_fold('tab', use_lc=False, use_tab=False, use_mix = True)
local_five_fold('tab', use_lc=False, use_tab=False, use_mix = True)
