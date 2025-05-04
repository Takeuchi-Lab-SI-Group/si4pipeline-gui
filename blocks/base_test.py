import pickle
import traceback

import numpy as np
import pandas as pd


class BaseTest:
    def __init__(self, st):
        self.st = st

    def perform_inference(self, block):
        print("perform inference!")

        plp_setting, pipeline = self.make_pipeline(block)
        print("pipeline:", pipeline)
        try:
            if self.st.session_state.dataset == "random":
                n, p = 100, 10
                rng = np.random.default_rng(0)
                X = rng.normal(size=(n, p))
                y = rng.normal(size=n)
                num_missing = rng.binomial(n, 0.03)
                mask = rng.choice(n, num_missing, replace=False)
                y[mask] = np.nan
                sigma = 1.0

                if plp_setting["tune_flag"]:
                    pipeline.tune(
                        X, y, num_folds=self.st.session_state.cv, random_state=0
                    )

                M, p_list = pipeline.inference(X, y, sigma)
                for each_feature, p_value in zip(M, p_list):
                    print(f"feature:{each_feature} p-value:{p_value:.3f}")
                results = []
                for each_feature, p_value in zip(M, p_list):
                    significance_status = (
                        "significant" if p_value <= 0.05 else "not significant"
                    )
                    result = {
                        "Feature": f"feature_{each_feature}",
                        "p-value": round(p_value, 3),
                        "Significance": significance_status,
                    }
                    results.append(result)
                results_df = pd.DataFrame(results)
                self.st.session_state["results_df"] = results_df

            elif self.st.session_state.dataset in [
                "prostate_cancer",
                "red_wine",
                "concrete",
                "abalone",
                "uploaded",
            ]:
                if self.st.session_state.dataset == "uploaded":
                    X, y, features = self.st.session_state.uploaded_dataset

                elif self.st.session_state.dataset == "prostate_cancer":
                    features = [
                        "lcavol",
                        "lweight",
                        "age",
                        "lbph",
                        "svi",
                        "lcp",
                        "gleason",
                        "pgg45",
                    ]
                    with open("dataset/prostate_cancer.pkl", "rb") as f:
                        X, y = pickle.load(f)

                elif self.st.session_state.dataset == "red_wine":
                    features = [
                        "fixed_acidity",
                        "volatile_acidity",
                        "citric_acid",
                        "residual_sugar",
                        "chlorides",
                        "free_sulfur_dioxide",
                        "total_sulfur_dioxide",
                        "density",
                        "pH",
                        "sulphates",
                        "alcohol",
                    ]
                    with open("dataset/red_wine.pkl", "rb") as f:
                        X, y = pickle.load(f)

                elif self.st.session_state.dataset == "concrete":
                    features = [
                        "cement",
                        "blast_furnace_slag",
                        "fly_ash",
                        "water",
                        "superplasticizer",
                        "coarse_aggregate",
                        "fine_aggregate",
                        "age",
                    ]
                    with open("dataset/concrete.pkl", "rb") as f:
                        X, y = pickle.load(f)

                elif self.st.session_state.dataset == "abalone":
                    features = [
                        "length",
                        "diameter",
                        "height",
                        "whole_weight",
                        "shucked_weight",
                        "viscera_weight",
                        "shell_weight",
                    ]
                    with open("dataset/abalone.pkl", "rb") as f:
                        X, y = pickle.load(f)

                if plp_setting["tune_flag"]:
                    pipeline.tune(
                        X, y, num_folds=self.st.session_state.cv, random_state=0
                    )

                M, p_list = pipeline.inference(X, y)

                print("Inference results are :\n")
                for each_feature, p_value in zip(M, p_list):
                    print(
                        f'{features[each_feature]}:\np-value is {p_value:.6f}, \
                            {"significant" if p_value <= 0.05 else "not significant"}\n'
                    )
                    significance_status = (
                        "significant" if p_value < 0.05 else "not significant"
                    )
                results = []
                for each_feature, p_value in zip(M, p_list):
                    significance_status = (
                        "significant" if p_value <= 0.05 else "not significant"
                    )
                    result = {
                        "Feature": features[each_feature],
                        "p-value": round(p_value, 6),
                        "Significance": significance_status,
                    }
                    results.append(result)
                results_df = pd.DataFrame(results)
                self.st.session_state["results_df"] = results_df
            else:
                raise Warning("unknown dataset")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    def make_pipeline(self2, self):
        raise NotImplementedError("make_pipeline method is not implemented")
