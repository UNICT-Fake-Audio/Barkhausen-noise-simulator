from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from matplotlib import colors as plt_colors
import numpy as np
from utils import BARKHAUSEN_LABELS, FEATURE_NAMES, ps, get_dataset_generators, get_metrics_generators
import shap

import joblib

dataset = {
        "train_path": "./train.csv",
        "test_path": "./test.csv",
        "drop_features": ["label", "file"],
}


def performance_generators(
    train_path: str,
    test_path: str,
    drop_features: list[str]
):
    _drop_features = drop_features
    x_train, _ = get_dataset_generators(
        train_path, _drop_features
    )

    # model = DecisionTreeClassifier().fit(x_train, y_train.ravel())
    # model = RandomForestClassifier(n_estimators=100).fit(x_train, y_train.ravel())
    # model = HistGradientBoostingClassifier().fit(x_train, y_train)
    model = joblib.load("models/hgbc_barkhausen.sav")
    # print("CLASSES", model.classes_)

    ### SHAP ###
    feature_names = FEATURE_NAMES

    shap_values = shap.TreeExplainer(model).shap_values(x_train)

    # custom color for bars
    BLUE = (0.145, 0, 1)
    YELLOW = (0.991, 1, 0)
    GREEN = (0, 0.515, 0)
    ORANGE = (1, 0.631, 0)
    colors = [BLUE, YELLOW, GREEN, ORANGE]
    class_inds = np.argsort(
        [-np.abs(shap_values[i]).mean() for i in range(len(shap_values))]
    )
    cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

    shap.summary_plot(
        shap_values,
        x_train,
        plot_type="bar",
        feature_names=feature_names,
        class_names=BARKHAUSEN_LABELS,
        color=cmap,
    )

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
