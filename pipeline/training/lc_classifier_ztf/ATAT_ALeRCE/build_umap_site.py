"""Build a static-HTML site of interactive UMAP plots from a pretrained backbone."""

from pathlib import Path

import numpy as np
import pandas as pd
import umap
import umap.plot
from bokeh.io import output_file, save

from src.exploringtools.ExplorePretraining import ExplorePretraining
from src.exploringtools.InitDataloader import InitDataLoader
from src.layers.transformer.ATAT import LightCurveTransformer
from src.utils.data.AlerceDictionaries import ZTF_TAXONOMY


PATH = "/home/magdalena/rpos/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/PRETRAIN/LC/20260409_scheduler_v3_claude_refactor_ema_1e3/"
FP_DATASET = "/home/magdalena/Desktop/sambashare/H5_files/BY_PARTITION/200_FF.h5"
DEVICE = "cuda:0"
TAXONOMY = ZTF_TAXONOMY
OUT_DIR = Path(__file__).parent / "umap_site"

COLORS_BY_CLASS = [
    "#ffacfa",  # agn
    "#D747CF",  # qso
    "#189a95",  # EA
    "#bb3e00",  # yso
    "#70df00",  # snia
    "#FE0912",  # cvnova
    "#009cff",  # rrlc
    "#0004ff",  # rscvn
    "#cd0066",  # blacar
    "#2F8B04",  # snii
    "#7a71f8",  # ebew
    "#3b3ee7",  # lpv
    "#41708f",  # cep
    "#00e3ff",  # rrlab
    "#1FA9FF",  # periodic other
    "#00e3ff",  # dsct
    "#458a00",  # snibc
    "#1E5A2E",  # slsn
    "#ff8000",  # tde
    "#FFC71F",  # sniib
    "#FFEC1F",  # sniin
    "#FFFFFF",  # microlensing
]

COLORS_BY_HIERARCHY = [
    "#fd31ee",  # agn
    "#fd31ee",  # qso
    "#00e3ff",  # EA
    "#fd31ee",  # yso
    "#1bda00",  # snia
    "#fd31ee",  # cvnova
    "#00e3ff",  # rrlc
    "#00e3ff",  # rscvn
    "#fd31ee",  # blacar
    "#1bda00",  # snii
    "#00e3ff",  # ebew
    "#00e3ff",  # lpv
    "#00e3ff",  # cep
    "#00e3ff",  # rrlab
    "#00e3ff",  # periodic other
    "#60fff3",  # dsct
    "#1bda00",  # snibc
    "#1bda00",  # slsn
    "#1bda00",  # tde
    "#1bda00",  # sniib
    "#1bda00",  # sniin
    "#1bda00",  # microlensing
]


def fit_umap():
    explore = ExplorePretraining(
        path_to_config_yaml=PATH, model=LightCurveTransformer, arg_key="lc"
    )
    od = explore.create_ordered_dict(
        remove_if_in_key_list=["projection", "transformer_tab", "classifier"],
        rename_keys=("model.", ""),
        checkpoint_name="pretrain_ckpt",
    )
    explore.load_backbone_weights(od)

    dl = InitDataLoader(
        update_dataset_path=FP_DATASET,
        update_batch_size=16,
        datamodule_args=explore.args.datamodule,
    )
    dl.init_validation_dataset()

    X_val, y_val, count_len = explore.predict(
        dl.validation_dataset, DEVICE, return_count_len=True
    )

    mapper = umap.UMAP(min_dist=0.5).fit(X_val)

    label_to_name = {value: key for key, value in TAXONOMY().items()}
    hover_data = pd.DataFrame(
        {
            "index": range(len(X_val)),
            "label": y_val,
            "detection_count": count_len.sum(axis=1),
            "class": [label_to_name[int(v)] for v in y_val],
        }
    )
    return mapper, y_val, count_len, hover_data


def render_interactive(mapper, labels, hover_data, color_key, out_path, title):
    p = umap.plot.interactive(
        mapper,
        labels=labels,
        hover_data=hover_data,
        point_size=5,
        tools=["pan", "wheel_zoom", "box_zoom", "save", "reset", "box_select"],
        width=1920,
        height=1080,
        color_key=color_key,
        background="black",
        alpha=0.9,
    )
    output_file(out_path, title=title)
    save(p)


def write_index(out_dir, entries):
    items = "\n".join(
        f'    <li><a href="{href}">{label}</a></li>' for href, label in entries
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>UMAP Visualizations</title>
  <style>
    body {{ background:#111; color:#eee; font-family: sans-serif; padding: 2rem; }}
    a {{ color:#00e3ff; }}
    li {{ margin: 0.5rem 0; }}
  </style>
</head>
<body>
  <h1>UMAP Visualizations</h1>
  <ul>
{items}
  </ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapper, y_val, count_len, hover_data = fit_umap()

    label_to_name = {value: key for key, value in TAXONOMY().items()}
    class_labels = np.array([label_to_name[int(v)] for v in y_val.astype(int)])

    render_interactive(
        mapper, class_labels, hover_data, COLORS_BY_CLASS,
        OUT_DIR / "by_class.html", "UMAP — by class",
    )
    render_interactive(
        mapper, class_labels, hover_data, COLORS_BY_HIERARCHY,
        OUT_DIR / "by_hierarchy.html", "UMAP — by hierarchy",
    )
    render_interactive(
        mapper, count_len.sum(axis=1), hover_data, None,
        OUT_DIR / "by_detection_count.html", "UMAP — by detection count",
    )

    write_index(
        OUT_DIR,
        [
            ("by_class.html", "By class"),
            ("by_hierarchy.html", "By hierarchy"),
            ("by_detection_count.html", "By detection count"),
        ],
    )

    print(f"Wrote site to {OUT_DIR}")
    print(f"Serve with:  python -m http.server --directory {OUT_DIR} 8000")


if __name__ == "__main__":
    main()
