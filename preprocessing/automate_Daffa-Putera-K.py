

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def parse_args():
    p = argparse.ArgumentParser(description="Automate preprocessing for House Prices dataset")
    p.add_argument("--raw_dir", type=str, default="namadataset_raw", help="Folder dataset mentah")
    p.add_argument("--out_dir", type=str, default="preprocessing/namadataset_preprocessing", help="Folder output preprocessing")
    p.add_argument("--target", type=str, default="SalePrice", help="Nama kolom target")
    p.add_argument("--test_size", type=float, default=0.2, help="Porsi validation split")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    return p.parse_args()


def load_raw(raw_dir: Path) -> pd.DataFrame:
    train_path = raw_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv tidak ditemukan di {train_path.resolve()}")
    return pd.read_csv(train_path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    feature_names = []
    feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

    return feature_names


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_dir)

    # Bersihkan duplikat (selaras dengan notebook yang “aman”)
    df = df.drop_duplicates()

    if args.target not in df.columns:
        raise ValueError(f"Kolom target '{args.target}' tidak ditemukan. Kolom tersedia: {list(df.columns)[:20]} ...")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Drop kolom Id (selaras best practice House Prices)
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Split train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Preprocess (fit di train, transform di valid)
    preprocessor = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_valid_p = preprocessor.transform(X_valid)

    feature_names = get_feature_names(preprocessor, X_train)

    X_train_p = pd.DataFrame(X_train_p, columns=feature_names)
    X_valid_p = pd.DataFrame(X_valid_p, columns=feature_names)

    # Save outputs
    X_train_p.to_csv(out_dir / "X_train_processed.csv", index=False)
    X_valid_p.to_csv(out_dir / "X_valid_processed.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False, header=True)
    y_valid.to_csv(out_dir / "y_valid.csv", index=False, header=True)

    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    print("✅ Preprocessing selesai. Output tersimpan di:", out_dir.resolve())
    print("Files:")
    for f in ["X_train_processed.csv", "X_valid_processed.csv", "y_train.csv", "y_valid.csv", "preprocessor.joblib"]:
        print("-", (out_dir / f).name)


if __name__ == "__main__":
    main()
