from .node_factory import create_node


def get_base_blocks(plp):
    mean_value_imputation = create_node(
        "mean_value_imputation",
        plp.mean_value_imputation,
        outputs=["y'"],
        inputs=["X", "y"],
    )
    soft_ipod = create_node(
        "soft_ipod",
        plp.soft_ipod,
        outputs=["O"],
        inputs=["X", "y"],
        options={"penalty coefficient": {"default": 0.015, "type": float}},
    )
    remove_outliers = create_node(
        "remove_outliers",
        plp.remove_outliers,
        outputs=["X", "y"],
        inputs=["X", "y", "O"],
    )
    marginal_screening = create_node(
        "marginal_screening",
        plp.marginal_screening,
        outputs=["M"],
        inputs=["X", "y"],
        options={"number of features": {"default": 5, "type": int}},
    )
    extract_features = create_node(
        "extract_features", plp.extract_features, outputs=["X"], inputs=["X", "M"]
    )
    stepwise_feature_selection = create_node(
        "stepwise_feature_selection",
        plp.stepwise_feature_selection,
        outputs=["M"],
        inputs=["X", "y"],
        options={"number of features": {"default": 3, "type": int}},
    )
    lasso = create_node(
        "lasso",
        plp.lasso,
        outputs=["M"],
        inputs=["X", "y"],
        options={"penalty coefficient": {"default": 0.08, "type": float}},
    )
    union = create_node("union", plp.union, outputs=["M"], inputs=["M1", "M2"])
    regression_imputation = create_node(
        "regression_imputation",
        plp.definite_regression_imputation,
        outputs=["y'"],
        inputs=["X", "y"],
    )
    cook_distance = create_node(
        "cook_distance",
        plp.cook_distance,
        outputs=["O"],
        inputs=["X", "y"],
        options={"penalty coefficient": {"default": 3.0, "type": float}},
    )
    intersection = create_node(
        "intersection", plp.intersection, outputs=["M"], inputs=["M1", "M2"]
    )

    return [
        mean_value_imputation,
        soft_ipod,
        remove_outliers,
        marginal_screening,
        extract_features,
        stepwise_feature_selection,
        lasso,
        union,
        regression_imputation,
        cook_distance,
        intersection,
    ]
