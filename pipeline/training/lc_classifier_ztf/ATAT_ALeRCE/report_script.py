# %%
import pandas as pd

from src.utils.data.AlerceDictionaries import ELASTICC_TAXONOMY, ZTF_TAXONOMY
from src.exploringtools.InitBackbone import InitBackbone
from src.exploringtools.InitClassifier import InitClassifier, ztf_order_classes
from src.exploringtools.InitDataloader import InitDataLoader

from src.exploringtools.InitCombinator import InitCombinator
from src.layers.classifiers.MultimodalClassifier import MultimodalClassifier
from src.layers.classifiers.HierClassifiery import Hier
from src.layers.transformer.ATAT import TabularTransformer, LightCurveTransformer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report



def summarize_classification_reports(reports):
    """
    Given a list of sklearn classification_report dictionaries (with output_dict=True),
    compute the mean and standard deviation across them and return a formatted string report.
    """
    # Convert each report to a DataFrame and collect them
    df_list = [pd.DataFrame(r).transpose() for r in reports]

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
    metrics = ['precision', 'recall', 'f1-score']

    for label in mean_df.index:
        if label == 'accuracy':
            # Accuracy is a single float, stored in the "precision" column
            acc_mean = mean_df.loc[label].get('precision', float('nan'))
            acc_std = std_df.loc[label].get('precision', float('nan'))
            report_str += f"{label:>12}  {acc_mean:.4f}$\\pm${acc_std:.4f}&\n"
        else:
            values = []
            for metric in metrics:
                mean_val = mean_df.loc[label].get(metric, float('nan'))
                std_val = std_df.loc[label].get(metric, float('nan'))
                values.append(f"{mean_val:.4f}$\\pm${std_val:.4f} &")
            support = mean_df.loc[label].get('support', float('nan'))
            report_str += f"{label:>12}  & " + "  ".join(values) + f"  {int(support):>5}" +'\\\ ' + '\n'
        top = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        '{} {} '.format(4*' & ',r'\\'),
        r'\midrule'
        '\n'
    ]
    bottom= [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Classification Report}',
        r'\label{tab:classification_report}',
        r'\end{table}'
    ]
    return '\n'.join(top) +  report_str+ '\n'.join(bottom)


##########


FP_DATASET = "/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/dataset.h5"
#FP_DATASET = "/home/mdelafuente/ORIGINAL/elasticc_dataset_update.h5"
FP_DATASET ='/home/mdelafuente/ZTF_SSL_Dataset/data/H5_files/BY_PARTITION/200_FF.h5'

DEVICE = 'cuda:0'
DATASET = 'test'
FP_DATASET ='/home/mdelafuente/ZTF_SSL_Dataset/data/H5_files/BY_PARTITION/200_FF.h5'
reports_1 = []
reports_2 = []




RANGE = 1
for i in range(RANGE):
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/64_128/LC_MD_FEAT/MM_class_64_128_{i}/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/64_128/LC_MD_FEAT/MM_class_64_128_{i}/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/AUGMENTED_FINAL/LC_MD_FEAT/MM_class_64_128_augmented_{i}/'
    #ATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/FINAL/LC_MD_FEAT/MM_class_64_128_{i}/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/FINAL/LC_MD_FEAT/MM_class_64_128_{i}_FROZEN/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/FINAL/LC_MD_FEAT/MM_class_v0000_MULTIMODAL_64128_0_{i}_FROZEN/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/SANITY_CHECK/LC_MD_FEAT/MM_class_sanity_check_{i}/'
    PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/SANITY_CHECK/LC_MD_FEAT/MM_class_newacc_{i}/'

    test = InitCombinator(path_to_config_yaml=PATH,
                    lc_model= LightCurveTransformer,
                    tab_model= TabularTransformer,
                    classifier = MultimodalClassifier,
                    classifier_args={'use_lc':False,
                                     'use_tab':False,
                                     'use_mix':True},
                    arg_key='tab',
                    device = DEVICE)
    dl = InitDataLoader(update_dataset_path=FP_DATASET,update_batch_size=32,datamodule_args=test.args.datamodule)
    dl.init_test_dataset()


    #val_preds, val_true =test.predict_for_n_samples(dl.test_dataset, DEVICE, n = 4) if DATASET == 'test' else test.predict(dl.validation_dataset, DEVICE)
    val_preds, val_true =test.predict(dl.test_dataset, DEVICE) if DATASET == 'test' else test.predict(dl.validation_dataset, DEVICE)
    #print(classification_report(val_true, np.argmax(val_preds, axis = -1), target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=False))
    #test.get_confusion_matrix(np.argmax(val_preds, axis = -1), val_true, dataset_type=DATASET, taxonomy = ZTF_TAXONOMY, plot_title= 'ZTF ATAT LC ONLY', order_classes=ztf_order_classes)
    reports_1.append(classification_report(val_true, np.argmax(val_preds, axis = -1), target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=True))


print(summarize_classification_reports(reports_1))
print("##################")
exit()

# %%


# %%

for i in range(RANGE):
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/FINAL/LC_MD_FEAT/MM_class_v0000_MULTIMODAL_64128_0_{i}_moredropout/'
    #PATH = f'/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/64_128/LC_MD_FEAT/MM_class_64_128_{i}/'
    test = InitCombinator(path_to_config_yaml=PATH,
                    lc_model= LightCurveTransformer,
                    tab_model= TabularTransformer,
                    classifier = MultimodalClassifier,
                    classifier_args={'use_lc':False,
                                     'use_tab':False,
                                     'use_mix':True},
                    arg_key='tab',
                    device = DEVICE)
    dl = InitDataLoader(update_dataset_path=FP_DATASET,update_batch_size=32,datamodule_args=test.args.datamodule)
    dl.init_test_dataset()
    val_preds, val_true =test.predict(dl.test_dataset, DEVICE) if DATASET == 'test' else test.predict(dl.validation_dataset, DEVICE)
    #print(classification_report(val_true, np.argmax(val_preds, axis = -1), target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=False))
    #test.get_confusion_matrix(np.argmax(val_preds, axis = -1), val_true, dataset_type=DATASET, taxonomy = ZTF_TAXONOMY, plot_title= 'ZTF ATAT LC ONLY', order_classes=ztf_order_classes)
    reports_2.append(classification_report(val_true, np.argmax(val_preds, axis = -1), target_names=list(ZTF_TAXONOMY().keys()),digits = 4, output_dict=True))


print(summarize_classification_reports(reports_1))
print('#####################')
print(summarize_classification_reports(reports_2))


# %%
from pprint import pprint
#pprint(reports_1[0])
import pandas as pd
metrics =[]

for rep in reports_1:

  metrics.append(pd.DataFrame(rep))


mean_stats_1 = pd.concat(metrics).reset_index().rename(columns = {'index':'metric'}).groupby('metric').describe()#.sort_values('level_1')#columns#.query('level_1 == std')
mean_stats_1 = mean_stats_1.T.reset_index().query('level_1 == "mean"')

for rep in reports_2:

  metrics.append(pd.DataFrame(rep))

mean_stats_2 = pd.concat(metrics).reset_index().rename(columns = {'index':'metric'}).groupby('metric').describe()#.sort_values('level_1')#columns#.query('level_1 == std')
mean_stats_2 = mean_stats_2.T.reset_index().query('level_1 == "mean"')



# %%
mean_collected = mean_stats_1.merge(mean_stats_2, on = ['level_0','level_1','support'],suffixes = ['_baseline','_pretrained'], how = 'left').drop(columns = 'level_1')
mean_collected = mean_collected[['level_0', 'precision_baseline', 'recall_baseline', 'f1-score_baseline',
        'precision_pretrained',
       'recall_pretrained','f1-score_pretrained', 'support']]
mean_collected['support'] =mean_collected['support'].astype(int)
mean_collected = mean_collected.round(4).rename(columns = {'level_0':'class'})#.drop(columns = 'metric')

# %%
from pprint import pprint
#pprint(reports_1[0])
import pandas as pd
metrics =[]

for rep in reports_1:

  metrics.append(pd.DataFrame(rep))


std_stats_1 = pd.concat(metrics).reset_index().rename(columns = {'index':'metric'}).groupby('metric').describe()#.sort_values('level_1')#columns#.query('level_1 == std')
std_stats_1 = std_stats_1.T.reset_index().query('level_1 == "std"')

for rep in reports_2:

  metrics.append(pd.DataFrame(rep))

std_stats_2 = pd.concat(metrics).reset_index().rename(columns = {'index':'metric'}).groupby('metric').describe()#.sort_values('level_1')#columns#.query('level_1 == std')
std_stats_2 = std_stats_2.T.reset_index().query('level_1 == "std"')



# %%
std_collected = std_stats_1.merge(std_stats_2, on = ['level_0','level_1','support'],suffixes = ['_baseline','_pretrained'], how = 'left').drop(columns = 'level_1')
std_collected = std_collected[['level_0', 'precision_baseline', 'recall_baseline', 'f1-score_baseline',
        'precision_pretrained',
       'recall_pretrained','f1-score_pretrained', 'support']]
std_collected['support'] =std_collected['support'].astype(int)
std_collected = std_collected.round(4).rename(columns = {'level_0':'class'})#.drop(columns = 'metric')

# %%
COLS = ['precision_baseline','recall_baseline','f1-score_baseline','precision_pretrained','recall_pretrained','f1-score_pretrained']
#formated = pd.concat([mean_collected['class'],"&"+mean_collected[COLS].astype(str) + "$\pm$" + std_collected[COLS].astype(str),"&" + mean_collected['support'].astype(str) + f'\ \\'], axis = 1)

# %%
formated = pd.concat([mean_collected['class'],mean_collected[COLS].astype(str) + "$\pm$" + std_collected[COLS].astype(str),mean_collected['support'].astype(str)], axis = 1)
print(formated.set_index('class').to_latex())


