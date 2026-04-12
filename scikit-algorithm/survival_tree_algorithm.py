import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sksurv.tree import SurvivalTree
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score


class SurvivalTreeAlgorithm:
    def __init__(
        self,
        survival_time_col,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=None,
        n_time_points=100,
    ):
        self.survival_time_col = survival_time_col
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_time_points = n_time_points

        self.preprocessor = None
        self.model = None
        self.y_train_struct_ = None
        self.feature_columns_ = None
        self.numeric_columns_ = None
        self.categorical_columns_ = None

    def _normalize_missing_values(self, df):
        return df.replace(
            {
                None: np.nan,
                "": np.nan,
                " ": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
                "na": np.nan,
                "null": np.nan,
                "None": np.nan,
            }
        )

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X = np.asarray(X, dtype=object)
            if X.ndim != 2:
                raise ValueError("X must be a 2D array or a pandas DataFrame.")
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

        df = self._normalize_missing_values(df)
        return df

    def _ensure_series(self, y, index=None, name="event"):
        if isinstance(y, pd.Series):
            s = y.copy()
        else:
            s = pd.Series(np.asarray(y), index=index, name=name)

        s = s.replace(
            {
                None: np.nan,
                "": np.nan,
                " ": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
                "na": np.nan,
                "null": np.nan,
                "None": np.nan,
            }
        )
        return s

    def _infer_column_types(self, X_df):
        numeric_columns = []
        categorical_columns = []

        for col in X_df.columns:
            series = X_df[col]
            non_missing = series.dropna()

            if len(non_missing) == 0:
                categorical_columns.append(col)
                continue

            converted = pd.to_numeric(non_missing, errors="coerce")

            if converted.notna().all():
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

        return numeric_columns, categorical_columns

    def _build_preprocessor(self, X_df):
        numeric_columns, categorical_columns = self._infer_column_types(X_df)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers = []
        if numeric_columns:
            transformers.append(("num", numeric_pipeline, numeric_columns))
        if categorical_columns:
            transformers.append(("cat", categorical_pipeline, categorical_columns))

        if not transformers:
            raise ValueError("No usable feature columns found.")

        self.numeric_columns_ = numeric_columns
        self.categorical_columns_ = categorical_columns

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _prepare_X_and_y_struct(self, X, y, fit=False):
        X_df = self._ensure_dataframe(X)

        if self.survival_time_col not in X_df.columns:
            raise ValueError(
                f"Column '{self.survival_time_col}' was not found in X."
            )

        y_series = self._ensure_series(y, index=X_df.index, name="event")

        if len(X_df) != len(y_series):
            raise ValueError("X and y must have the same number of rows.")

        time_series = pd.to_numeric(X_df[self.survival_time_col], errors="coerce")

        if time_series.isna().any():
            bad_rows = time_series[time_series.isna()].index.tolist()[:10]
            raise ValueError(
                f"Column '{self.survival_time_col}' contains invalid or missing times. "
                f"Example row indices: {bad_rows}"
            )

        if y_series.isna().any():
            bad_rows = y_series[y_series.isna()].index.tolist()[:10]
            raise ValueError(
                f"y contains missing event values. Example row indices: {bad_rows}"
            )

        X_features = X_df.drop(columns=[self.survival_time_col]).copy()

        if fit:
            self.feature_columns_ = list(X_features.columns)
        else:
            if self.feature_columns_ is None:
                raise ValueError("The model has not been fitted yet.")

            missing_cols = [c for c in self.feature_columns_ if c not in X_features.columns]
            extra_cols = [c for c in X_features.columns if c not in self.feature_columns_]

            if missing_cols:
                raise ValueError(
                    f"X is missing feature columns used during fit: {missing_cols}"
                )

            # ignorujemy nadmiarowe kolumny, ale trzymamy kolejność z treningu
            X_features = X_features[self.feature_columns_]

        # event -> bool
        if y_series.dtype == bool:
            event = y_series.to_numpy(dtype=bool)
        else:
            event = y_series.astype(str).str.strip().str.lower().map(
                {
                    "1": True,
                    "true": True,
                    "t": True,
                    "yes": True,
                    "y": True,
                    "0": False,
                    "false": False,
                    "f": False,
                    "no": False,
                    "n": False,
                }
            )

            if event.isna().any():
                # fallback dla typów liczbowych
                try:
                    event = y_series.astype(int).astype(bool)
                except Exception as e:
                    bad_rows = y_series[event.isna()].index.tolist()[:10]
                    raise ValueError(
                        f"Could not convert y to boolean event indicator. "
                        f"Example problematic row indices: {bad_rows}"
                    ) from e
            else:
                event = event.to_numpy(dtype=bool)

        time = time_series.to_numpy(dtype=float)

        y_struct = Surv.from_arrays(event=event, time=time)
        return X_features, y_struct

    def _fit_transform_X(self, X_features):
        self.preprocessor = self._build_preprocessor(X_features)
        X_transformed = self.preprocessor.fit_transform(X_features)
        return X_transformed

    def _transform_X(self, X_features):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X_features)

    def fit(self, X, y):
        X_features, y_struct = self._prepare_X_and_y_struct(X, y, fit=True)
        X_transformed = self._fit_transform_X(X_features)

        self.model = SurvivalTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        self.model.fit(X_transformed, y_struct)
        self.y_train_struct_ = y_struct
        return self

    def predict_survival_function(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        dummy_y = pd.Series(np.zeros(len(X), dtype=int))
        X_features, _ = self._prepare_X_and_y_struct(X, dummy_y, fit=False)
        X_transformed = self._transform_X(X_features)

        return self.model.predict_survival_function(X_transformed)

    def integrated_brier_score(self, X, y, times=None):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        if self.y_train_struct_ is None:
            raise ValueError("Training survival data is missing.")

        X_features, y_test_struct = self._prepare_X_and_y_struct(X, y, fit=False)
        X_transformed = self._transform_X(X_features)

        train_times = self.y_train_struct_["time"].astype(float)
        test_times = y_test_struct["time"].astype(float)

        if times is None:
            lower = max(np.min(train_times), np.min(test_times))
            upper = min(np.max(train_times), np.max(test_times))

            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    "Could not automatically determine a valid time grid for integrated_brier_score."
                )

            upper = np.nextafter(upper, lower)
            times = np.linspace(lower, upper, self.n_time_points)
        else:
            times = np.asarray(times, dtype=float)

        surv_fns = self.model.predict_survival_function(X_transformed)
        surv_probs = np.vstack([fn(times) for fn in surv_fns])

        return integrated_brier_score(
            self.y_train_struct_,
            y_test_struct,
            surv_probs,
            times,
        )

    def get_rules_text(self, simplify_onehot=True):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")

        tree = self.model.tree_
        feature_names = list(self.preprocessor.get_feature_names_out())
        rules = []

        def format_condition(feature_name, operator, threshold):
            threshold_txt = f"{threshold:.6g}"

            if not simplify_onehot:
                return f"{feature_name} {operator} {threshold_txt}"

            if feature_name.startswith("cat__"):
                raw_name = feature_name[len("cat__"):]

                if "_" in raw_name and abs(threshold - 0.5) < 1e-12:
                    original_col, category = raw_name.split("_", 1)
                    if operator == ">":
                        return f"{original_col} = {category}"
                    if operator == "<=":
                        return f"{original_col} != {category}"

            if feature_name.startswith("num__"):
                feature_name = feature_name[len("num__"):]

            return f"{feature_name} {operator} {threshold_txt}"

        def recurse(node_id, conditions):
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            is_leaf = left_child == -1 and right_child == -1

            if is_leaf:
                rules.append(" and ".join(conditions) if conditions else "TRUE")
                return

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]

            recurse(
                left_child,
                conditions + [format_condition(feature_name, "<=", threshold)]
            )
            recurse(
                right_child,
                conditions + [format_condition(feature_name, ">", threshold)]
            )

        recurse(0, [])
        return rules
    

    def get_rules_statistics(self, simplify_onehot=True):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        rules = self.get_rules_text(simplify_onehot=simplify_onehot)

        lengths = []
        for rule in rules:
            if rule.strip() == "TRUE":
                lengths.append(0)
            else:
                lengths.append(len(rule.split(" and ")))

        liczba_regul = len(rules)
        suma_warunkow = int(sum(lengths))
        srednia_dlugosc_reguly = (
            float(suma_warunkow / liczba_regul) if liczba_regul > 0 else 0.0
        )

        return {
            "liczba_regul": liczba_regul,
            "srednia_dlugosc_reguly": srednia_dlugosc_reguly,
            "suma_warunkow": suma_warunkow,
        }