"""Genetic Algorithm wrapper feature selection for SVM credit risk analysis.

This script:
1. Loads and preprocesses German credit data.
2. Computes baseline 10-fold CV accuracy using all engineered features.
3. Runs a GA (DEAP) to select an optimal feature subset.
4. Reports best subset, feature reduction, and final CV accuracy.

GA hyperparameters:
- Population size: 40 (tuned for stronger search diversity)
- Generations: 20
- Crossover probability: 0.6
- Mutation probability: 0.03
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from deap import algorithms, base, creator, tools
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DEAP is required for this script. Install it with: pip install deap"
    ) from exc


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("german_credit_data.csv")

NUMERICAL_FEATURES = [
    "Age",
    "Job",
    "Checking account",
    "Credit amount",
    "Duration",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Purpose",
]

TARGET_CANDIDATES = [
    "Credit Risk",
    "Risk",
    "credit_risk",
    "risk",
    "target",
    "class",
    "label",
    "y",
]

# User-tuned GA search budget for stronger exploration.
POPULATION_SIZE = 40
N_GENERATIONS = 20
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.08

RANDOM_SEED = 42
PENALTY_WEIGHT = 0.0
ELITE_SIZE = 4
IMMIGRANT_RATIO = 0.10
UCI_GERMAN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def _make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names for consistent matching."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _drop_auto_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop common auto-generated index columns if present."""
    candidates = ["Unnamed: 0", "index", "Index"]
    return df.drop(columns=[c for c in candidates if c in df.columns], errors="ignore")


def _encode_checking_account_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Checking account' to ordinal numeric if it is stored as text.

    This follows the user-specified requirement to treat it as numerical.
    """
    if "Checking account" not in df.columns:
        return df

    out = df.copy()
    col = out["Checking account"]

    if pd.api.types.is_numeric_dtype(col):
        return out

    mapping = {
        "na": 0,
        "none": 0,
        "little": 1,
        "moderate": 2,
        "rich": 3,
        "quite rich": 4,
    }
    out["Checking account"] = (
        col.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "na", "": "na"})
        .map(mapping)
        .fillna(0)
    )
    return out


def _split_feature_types(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical groups."""
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Build sklearn preprocessor for numeric + categorical columns."""
    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, list(numeric_cols)))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_ohe()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_cols)))

    if not transformers:
        raise ValueError("No valid feature columns found after preprocessing.")

    return ColumnTransformer(transformers=transformers)


def _find_target_column(columns: Sequence[str]) -> str | None:
    """Find target column from known candidate names (case-insensitive)."""
    col_map = {c.lower(): c for c in columns}
    for cand in TARGET_CANDIDATES:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def _encode_target_binary(y: pd.Series) -> np.ndarray:
    """Convert target to binary array (0/1)."""
    if pd.api.types.is_numeric_dtype(y):
        values = y.to_numpy()
        unique = np.unique(values[~pd.isna(values)])
        if set(unique).issubset({0, 1}):
            return values.astype(int)

    labels = y.astype(str).str.strip().str.lower()
    good_tokens = {"good", "1", "yes", "true", "approved"}
    bad_tokens = {"bad", "0", "no", "false", "rejected"}

    parsed = []
    for v in labels:
        if v in good_tokens:
            parsed.append(1)
        elif v in bad_tokens:
            parsed.append(0)
        else:
            raise ValueError(
                f"Unrecognized target label '{v}'. "
                "Use binary labels such as Good/Bad or 1/0."
            )
    return np.array(parsed, dtype=int)


def _build_proxy_target(df: pd.DataFrame) -> np.ndarray:
    """Create a proxy target when no explicit risk label exists.

    The Kaggle readable file commonly omits the original UCI risk label.
    This proxy is for workflow demonstration only and should be replaced by
    the real target for research conclusions.
    """
    needed = ["Credit amount", "Duration", "Checking account"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "Target column missing and proxy target cannot be built because "
            f"these columns are missing: {missing}"
        )

    credit = pd.to_numeric(df["Credit amount"], errors="coerce").fillna(df["Credit amount"].median())
    duration = pd.to_numeric(df["Duration"], errors="coerce").fillna(df["Duration"].median())
    checking = pd.to_numeric(df["Checking account"], errors="coerce").fillna(0)

    credit_z = (credit - credit.mean()) / (credit.std(ddof=0) + 1e-9)
    duration_z = (duration - duration.mean()) / (duration.std(ddof=0) + 1e-9)
    low_checking = (checking <= 1).astype(float)

    risk_score = 0.45 * credit_z + 0.35 * duration_z + 0.20 * low_checking
    threshold = risk_score.median()
    return (risk_score >= threshold).astype(int).to_numpy()


def _try_load_uci_risk_labels(expected_rows: int) -> np.ndarray | None:
    """Try downloading official UCI German credit labels.

    Returns None if download/parsing fails or row count mismatch occurs.
    """
    try:
        with urlopen(UCI_GERMAN_DATA_URL, timeout=10) as response:
            content = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if len(lines) != expected_rows:
        return None

    labels: List[int] = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            return None
        raw = parts[-1]
        if raw == "1":
            labels.append(1)
        elif raw == "2":
            labels.append(0)
        else:
            return None

    return np.array(labels, dtype=int)


def _prepare_german_dataset(
    df: pd.DataFrame,
    allow_proxy_target: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Preprocess German credit dataset with requested fixed feature treatment."""
    required = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    target_col = _find_target_column(df.columns)
    if target_col is None:
        if not allow_proxy_target:
            raise ValueError(
                "Target column not found. Expected one of: "
                f"{TARGET_CANDIDATES}."
            )
        uci_labels = _try_load_uci_risk_labels(expected_rows=len(df))
        if uci_labels is not None:
            y = uci_labels
            target_source = "uci_download"
        else:
            y = _build_proxy_target(df)
            target_source = "proxy"
    else:
        y = _encode_target_binary(df[target_col])
        target_source = f"column:{target_col}"

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    preprocessor = _build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return np.asarray(X_processed), y, feature_names, target_source


def _prepare_generic_dataset(
    df: pd.DataFrame,
    allow_proxy_target: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Preprocess a general credit-risk dataset with auto feature typing."""
    target_col = _find_target_column(df.columns)
    if target_col is None:
        if not allow_proxy_target:
            raise ValueError(
                "Target column not found. Expected one of: "
                f"{TARGET_CANDIDATES}."
            )
        y = _build_proxy_target(df)
        target_source = "proxy"
    else:
        y = _encode_target_binary(df[target_col])
        target_source = f"column:{target_col}"

    feature_columns = [c for c in df.columns if c != target_col]

    # Drop high-cardinality ID-like columns to avoid meaningless one-hot expansion.
    filtered_features: List[str] = []
    for col in feature_columns:
        lower = col.lower().strip()
        unique_ratio = df[col].nunique(dropna=False) / max(len(df), 1)
        is_id_like = ("id" in lower) and unique_ratio > 0.95
        if not is_id_like:
            filtered_features.append(col)

    if not filtered_features:
        raise ValueError("No usable feature columns found after removing ID-like columns.")

    numeric_cols, categorical_cols = _split_feature_types(df, filtered_features)
    X = df[filtered_features].copy()
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return np.asarray(X_processed), y, feature_names, target_source


def load_and_preprocess_data(
    csv_path: Path,
    allow_proxy_target: bool = True,
    dataset_mode: Literal["german", "generic"] = "german",
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load CSV and return preprocessed matrix, target vector, feature names, and target source."""
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    df = _drop_auto_index_columns(df)
    df = _encode_checking_account_if_needed(df)

    if dataset_mode == "german":
        return _prepare_german_dataset(df=df, allow_proxy_target=allow_proxy_target)
    if dataset_mode == "generic":
        return _prepare_generic_dataset(df=df, allow_proxy_target=allow_proxy_target)

    raise ValueError("Invalid dataset_mode. Use 'german' or 'generic'.")


def cross_val_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Compute 10-fold CV accuracy with an SVM classifier."""
    svm = SVC(kernel="rbf", gamma="scale", random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(svm, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores))


def run_ga_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, float]:
    """Run GA wrapper feature selection and return best chromosome and true accuracy."""
    n_features = X.shape[1]
    cache_penalized: Dict[Tuple[int, ...], float] = {}
    cache_raw_acc: Dict[Tuple[int, ...], float] = {}

    def evaluate_raw(mask_tuple: Tuple[int, ...]) -> float:
        if mask_tuple in cache_raw_acc:
            return cache_raw_acc[mask_tuple]

        mask = np.array(mask_tuple, dtype=bool)
        if not np.any(mask):
            cache_raw_acc[mask_tuple] = 0.0
            return 0.0

        score = cross_val_accuracy(X[:, mask], y)
        cache_raw_acc[mask_tuple] = score
        return score

    def fitness(individual: List[int]) -> Tuple[float]:
        key = tuple(int(g) for g in individual)
        if key in cache_penalized:
            return (cache_penalized[key],)

        raw_acc = evaluate_raw(key)
        selected_ratio = sum(key) / n_features
        penalized = raw_acc - PENALTY_WEIGHT * selected_ratio
        cache_penalized[key] = penalized
        return (penalized,)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    population = toolbox.population(n=POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)

    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    hall_of_fame.update(population)
    initial_best = hall_of_fame[0]
    print(
        "Generation 0 | Best CV accuracy: "
        f"{evaluate_raw(tuple(int(g) for g in initial_best)):.4f}"
    )

    for generation in range(1, N_GENERATIONS + 1):
        # Preserve top individuals so stochastic operators do not lose strong solutions.
        elites = tools.selBest(population, ELITE_SIZE)
        elite_copies = [creator.Individual(ind) for ind in elites]
        for elite_copy, elite in zip(elite_copies, elites):
            elite_copy.fitness.values = elite.fitness.values

        offspring = algorithms.varAnd(
            population,
            toolbox,
            cxpb=CROSSOVER_PROB,
            mutpb=MUTATION_PROB,
        )

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        # Diversity injection: replace a small fraction with fresh random individuals.
        immigrant_count = max(1, int(len(offspring) * IMMIGRANT_RATIO))
        immigrant_indices = random.sample(range(len(offspring)), immigrant_count)
        for idx in immigrant_indices:
            immigrant = toolbox.individual()
            immigrant.fitness.values = toolbox.evaluate(immigrant)
            offspring[idx] = immigrant

        population = toolbox.select(offspring, k=len(population) - ELITE_SIZE)
        population.extend(elite_copies)
        hall_of_fame.update(population)

        best = hall_of_fame[0]
        best_raw_acc = evaluate_raw(tuple(int(g) for g in best))
        print(f"Generation {generation:2d} | Best CV accuracy: {best_raw_acc:.4f}")

    best_chromosome = np.array(hall_of_fame[0], dtype=int)
    best_accuracy = evaluate_raw(tuple(int(g) for g in best_chromosome))

    # Safety fallback: avoid returning empty feature set.
    if not np.any(best_chromosome):
        best_chromosome = np.ones(len(feature_names), dtype=int)
        best_accuracy = cross_val_accuracy(X, y)

    return best_chromosome, best_accuracy


def run_traditional_selectkbest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, float, int]:
    """Run a traditional filter method (SelectKBest + mutual information).

    The best k is selected by the same 10-fold CV objective as the GA method.
    """
    n_features = X.shape[1]
    best_accuracy = -1.0
    best_k = n_features
    best_mask = np.ones(n_features, dtype=int)

    for k in range(1, n_features + 1):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_k = selector.fit_transform(X, y)
        acc = cross_val_accuracy(X_k, y)

        # Tie-break on smaller k for a more compact subset.
        if acc > best_accuracy + 1e-12 or (abs(acc - best_accuracy) <= 1e-12 and k < best_k):
            best_accuracy = acc
            best_k = k
            best_mask = selector.get_support().astype(int)

    return best_mask, float(best_accuracy), int(best_k)


def build_comparison_table(
    total_features: int,
    baseline_accuracy: float,
    traditional_accuracy: float,
    traditional_selected_count: int,
    ga_accuracy: float,
    ga_selected_count: int,
) -> pd.DataFrame:
    """Build side-by-side comparison of baseline, traditional, and GA methods."""
    rows = [
        {
            "Method": "Baseline (All Features)",
            "Selected Features": total_features,
            "10-fold CV Accuracy": baseline_accuracy,
            "Accuracy Gain vs Baseline": 0.0,
        },
        {
            "Method": "Traditional (SelectKBest)",
            "Selected Features": traditional_selected_count,
            "10-fold CV Accuracy": traditional_accuracy,
            "Accuracy Gain vs Baseline": traditional_accuracy - baseline_accuracy,
        },
        {
            "Method": "GA Wrapper (SVM Fitness)",
            "Selected Features": ga_selected_count,
            "10-fold CV Accuracy": ga_accuracy,
            "Accuracy Gain vs Baseline": ga_accuracy - baseline_accuracy,
        },
    ]
    return pd.DataFrame(rows)


def report_results(
    feature_names: Sequence[str],
    best_chromosome: np.ndarray,
    baseline_accuracy: float,
    final_accuracy: float,
) -> None:
    """Print final summary in a user-friendly format."""
    selected_indices = np.where(best_chromosome == 1)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    total_features = len(feature_names)
    selected_count = len(selected_features)
    reduced_count = total_features - selected_count

    print("\n" + "=" * 72)
    print("GA Wrapper Feature Selection Summary")
    print("=" * 72)
    print(f"Baseline (all {total_features} features) 10-fold CV accuracy: {baseline_accuracy:.4f}")
    print(f"Optimized subset 10-fold CV accuracy:                  {final_accuracy:.4f}")
    print(f"Features selected: {selected_count}/{total_features}")
    print(f"Features reduced:  {reduced_count}")
    print("\nSelected feature subset:")
    for feat in selected_features:
        print(f"- {feat}")


def run_experiment(
    csv_path: Path,
    allow_proxy_target: bool = True,
    dataset_mode: Literal["german", "generic"] = "german",
) -> Dict[str, object]:
    """Run the full GA feature-selection pipeline and return structured outputs."""
    X, y, feature_names, target_source = load_and_preprocess_data(
        csv_path=csv_path,
        allow_proxy_target=allow_proxy_target,
        dataset_mode=dataset_mode,
    )

    print("Computing baseline SVM performance (all features)...")
    baseline_accuracy = cross_val_accuracy(X, y)

    print("Running traditional benchmark (SelectKBest + SVM)...")
    traditional_mask, traditional_accuracy, best_k = run_traditional_selectkbest(X, y, feature_names)

    print("Running GA wrapper feature selection...")
    best_chromosome, final_accuracy = run_ga_feature_selection(X, y, feature_names)

    selected_indices = np.where(best_chromosome == 1)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    traditional_indices = np.where(traditional_mask == 1)[0]
    traditional_features = [feature_names[i] for i in traditional_indices]

    comparison_df = build_comparison_table(
        total_features=len(feature_names),
        baseline_accuracy=baseline_accuracy,
        traditional_accuracy=traditional_accuracy,
        traditional_selected_count=len(traditional_features),
        ga_accuracy=final_accuracy,
        ga_selected_count=len(selected_features),
    )

    return {
        "target_source": target_source,
        "baseline_accuracy": baseline_accuracy,
        "traditional_accuracy": traditional_accuracy,
        "traditional_best_k": best_k,
        "traditional_selected_features": traditional_features,
        "traditional_selected_count": len(traditional_features),
        "final_accuracy": final_accuracy,
        "feature_names": feature_names,
        "selected_features": selected_features,
        "selected_count": len(selected_features),
        "total_count": len(feature_names),
        "reduced_count": len(feature_names) - len(selected_features),
        "comparison_table": comparison_df.to_dict(orient="records"),
    }


def main() -> None:
    """Main execution flow."""
    print("Loading and preprocessing data...")
    X, y, feature_names, target_source = load_and_preprocess_data(DATA_PATH)

    if target_source == "proxy":
        print(
            "WARNING: No explicit target column found. "
            "Using a proxy risk label derived from credit attributes."
        )
    else:
        print(f"Using target from {target_source}.")

    print("Computing baseline SVM performance (all features)...")
    baseline_accuracy = cross_val_accuracy(X, y)
    print(f"Baseline 10-fold CV accuracy: {baseline_accuracy:.4f}")

    print("\nRunning traditional benchmark (SelectKBest + SVM)...")
    traditional_mask, traditional_accuracy, best_k = run_traditional_selectkbest(X, y, feature_names)
    print(f"Traditional best-k (k={best_k}) 10-fold CV accuracy: {traditional_accuracy:.4f}")

    print("\nRunning Genetic Algorithm for feature selection...")
    best_chromosome, final_accuracy = run_ga_feature_selection(X, y, feature_names)

    report_results(feature_names, best_chromosome, baseline_accuracy, final_accuracy)

    traditional_selected = int(np.sum(traditional_mask))
    ga_selected = int(np.sum(best_chromosome))
    comparison_df = build_comparison_table(
        total_features=len(feature_names),
        baseline_accuracy=baseline_accuracy,
        traditional_accuracy=traditional_accuracy,
        traditional_selected_count=traditional_selected,
        ga_accuracy=final_accuracy,
        ga_selected_count=ga_selected,
    )

    print("\n" + "=" * 72)
    print("Side-by-Side Method Comparison")
    print("=" * 72)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
