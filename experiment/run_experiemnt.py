import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib import colors as plt_colors
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils import (
    BARKHAUSEN_LABELS,
    FEATURE_NAMES,
    get_dataset_generators,
    get_metrics_generators,
    ps,
)

dataset = {
    "train_path": "./train.csv",
    "test_path": "./test.csv",
    "drop_features": ["label", "file"],
}


def performance_generators(train_path: str, test_path: str, drop_features: list[str]):
    _drop_features = drop_features
    x_train, _ = get_dataset_generators(train_path, _drop_features)

    # model = DecisionTreeClassifier().fit(x_train, y_train.ravel())
    # model = RandomForestClassifier(n_estimators=100).fit(x_train, y_train.ravel())
    # model = HistGradientBoostingClassifier().fit(x_train, y_train)
    model = joblib.load("models/hgbc_barkhausen.sav")
    # print("CLASSES", model.classes_)

    ### SHAP ###
    feature_names = FEATURE_NAMES

    shap_values = shap.TreeExplainer(model).shap_values(x_train)

    # newer shap returns a 3D array (n_samples, n_features, n_classes);
    # convert to a list of per-class 2D arrays for compatibility
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    # custom color for bars
    BLUE = (0.145, 0, 1)
    YELLOW = (0.991, 1, 0)
    # GREY = (0.5, 0.5, 0.5)
    GREEN = (0, 0.515, 0)
    ORANGE = (1, 0.631, 0)
    colors = [BLUE, GREEN, ORANGE, YELLOW]
    class_inds = np.argsort(
        [-np.abs(shap_values[i]).mean() for i in range(len(shap_values))]
    )
    cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

    shap.summary_plot(
        shap_values,
        x_train,
        plot_type="vertical_bar",
        feature_names=feature_names,
        class_names=BARKHAUSEN_LABELS,
        color=cmap,
        show=False,
    )
    ax = plt.gca()
    ax.invert_xaxis()
    ax.axvline(x=ax.get_xlim()[0], color="#999999", linewidth=1.5, zorder=-1)

    LEGEND_FONT_SIZE = 30
    AXIS_LABEL_FONT_SIZE = 18

    ax.tick_params(axis="x", labelsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=AXIS_LABEL_FONT_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    legend = ax.get_legend()

    if legend:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_FONT_SIZE)
    plt.show()

    ### END SHAP ###

    _drop_features = drop_features
    x, y = get_dataset_generators(test_path, _drop_features)
    y = y.ravel()

    # joblib.dump(model, "models/hgbc___barkhausen.sav")

    return get_metrics_generators(model, x, y, True, True)


print("| model                      | acc_c   |")
print("| -------------------------- | ------- |")

accuracies = []


conf_matrix, normalized_conf_matrix, accuracy_per_class, accuracy_overall = (
    performance_generators(
        dataset["train_path"],
        dataset["test_path"],
        dataset["drop_features"],
    )
)
print(f"| HGBC \t\t| {ps(accuracy_per_class)} \t\t|")
accuracies.append(ps(accuracy_per_class * 100, 5))
