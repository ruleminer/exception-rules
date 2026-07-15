import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DecisionTreeAlgorithm:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.preprocessor = None
        self.model = None
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

    def _ensure_series(self, y, index=None, name="target"):
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

    def _prepare_X_and_y(self, X, y=None, fit=False):
        X_df = self._ensure_dataframe(X)

        if y is not None:
            y_series = self._ensure_series(y, index=X_df.index, name="target")

            if len(X_df) != len(y_series):
                raise ValueError("X and y must have the same number of rows.")

            y_numeric = pd.to_numeric(y_series, errors="coerce")

            if y_numeric.isna().any():
                bad_rows = y_numeric[y_numeric.isna()].index.tolist()[:10]
                raise ValueError(
                    f"y contains invalid or missing numeric target values. "
                    f"Example row indices: {bad_rows}"
                )
        else:
            y_numeric = None

        X_features = X_df.copy()

        if fit:
            self.feature_columns_ = list(X_features.columns)
        else:
            if self.feature_columns_ is None:
                raise ValueError("The model has not been fitted yet.")

            missing_cols = [c for c in self.feature_columns_ if c not in X_features.columns]

            if missing_cols:
                raise ValueError(
                    f"X is missing feature columns used during fit: {missing_cols}"
                )

            # ignorujemy nadmiarowe kolumny, ale trzymamy kolejność z treningu
            X_features = X_features[self.feature_columns_]

        return X_features, None if y_numeric is None else y_numeric.to_numpy(dtype=float)

    def _fit_transform_X(self, X_features):
        self.preprocessor = self._build_preprocessor(X_features)
        X_transformed = self.preprocessor.fit_transform(X_features)
        return X_transformed

    def _transform_X(self, X_features):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X_features)

    def fit(self, X, y):
        X_features, y_array = self._prepare_X_and_y(X, y, fit=True)
        X_transformed = self._fit_transform_X(X_features)

        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        self.model.fit(X_transformed, y_array)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        X_features, _ = self._prepare_X_and_y(X, y=None, fit=False)
        X_transformed = self._transform_X(X_features)

        return self.model.predict(X_transformed)

    def score(self, X, y, metric="rmse"):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        X_features, y_true = self._prepare_X_and_y(X, y, fit=False)
        X_transformed = self._transform_X(X_features)
        y_pred = self.model.predict(X_transformed)

        metric = str(metric).strip().lower()

        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mse":
            return float(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            return float(r2_score(y_true, y_pred))
        else:
            raise ValueError(
                "Unsupported metric. Use one of: 'rmse', 'mse', 'mae', 'r2'."
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