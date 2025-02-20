import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

BARKHAUSEN_LABELS =  ['default', 'gamma_10', 'noise05', 'k_10']

FEATURE_NAMES = [
    "spectrum",
    "mean_frequency",
    "peak_frequency",
    "frequencies_std",
    "amplitudes_cum_sum",
    "mode_frequency",
    "median_frequency",
    "frequencies_q25",
    "frequencies_q75",
    "iqr",
    "freqs_skewness",
    "freqs_kurtosis",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_centroid",
    "spectral_spread",
    "spectral_rolloff",
    "energy",
    "rms",
    "zcr",
    "spectral_mean",
    "spectral_rms",
    "spectral_std",
    "spectral_variance",
    "meanfun",
    "minfun",
    "maxfun",
    "meandom",
    "mindom",
    "maxdom",
    "dfrange",
    "modindex",
    "bit_rate",
    "signal",
    "mfcc",
    "imfcc",
    "bfcc",
    "lfcc",
    "lpc",
    "lpcc",
    "msrcc",
    "ngcc",
    "psrcc",
    "plp",
    "rplp",
    "gfcc",
]


def get_dataset_generators(
    dataset_path: str,
    drop_feature: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_path)

    features = df.keys()
    features = features.drop(drop_feature)

    x: np.ndarray = df.loc[:, features].values
    y: np.ndarray = df.loc[:, ["label"]].values

    return x, y


def get_metrics_generators(
    model,
    x,
    y,
    print_metrics=False,
    show_graph=False,
):
    y_predicted = model.predict(x)
    pred = model.predict_proba(x)

    # print("model classes", model.classes_)

    labels = model.classes_
    display_labels = BARKHAUSEN_LABELS

    conf_matrix = confusion_matrix(
        y,
        y_predicted,
        labels=model.classes_,  # only for asvspoof model.classes_ and not just labels
        normalize="true",
    )
    accuracy_per_class = balanced_accuracy_score(y, y_predicted)
    accuracy_overall = accuracy_score(y, y_predicted)

    normalized_conf_matrix: list[float] = []

    if print_metrics:
        print("acc_c: \t" + str(accuracy_per_class))
        print("acc_o: \t" + str(accuracy_overall))
        # print("\n conf_matrix: \n" + str(conf_matrix))
        # print("\n normalized conf_matrix: \t" + str(normalized_conf_matrix))

    if show_graph:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=display_labels if display_labels else labels,
        )
        disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
        for labels in disp.text_.ravel():
            labels.set_fontsize(25)
        plt.show()

    return conf_matrix, normalized_conf_matrix, accuracy_per_class, accuracy_overall

# parse_string
def ps(s: str, digits=7) -> str:
    return str(s)[:digits]
