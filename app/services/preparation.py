from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from app.utils.logger import get_logger

# Our goal here is to:
# - Load the raw data safely and validate the schema
# - Generate a single multi-class target from three target columns (seller_number, customer_number, main_account)
#   (to learn the business rule directly)
# - Clean missing values and perform type conversions
# - Text (TF-IDF word+char) + field-focused text features (CustomTextFeatures),
#   categorical (OneHotEncoder), numerical (StandardScaler) consistent preprocessing build a pipeline
# - Case-appropriate train/test split (200 records: 150/50; otherwise %25 test)
# - Stratify to maintain class balance

logger = get_logger()


FEATURE_COLUMNS: List[str] = [
    "company_code",
    "document_number",
    "description",
    "payment_type",
    "amount",
    "currency_code",
    "transaction_type",
]

TARGET_COLUMNS: List[str] = ["seller_number", "customer_number", "main_account"]

# Advanced text normalization for Turkish text
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = re.sub(r"https?://\S+", " <URL> ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b", " <EMAIL> ", s)
    s = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " <DATE> ", s)
    s = re.sub(r"\b\d+[.,]?\d*\b", " <NUM> ", s)
    s = re.sub(r"\b(try|tl|usd|eur|gbp)\b", " <CUR> ", s)
    s = re.sub(r"[^a-z0-9<>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# To add simple but effective features that TF-IDF can't capture 
# (e.g. tax keywords, alphanumeric code pattern, description length, etc.) to the model
class CustomTextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            df_X = pd.DataFrame({'description': X})
        elif hasattr(X, 'columns'):
            df_X = X.copy()
        else:
            df_X = pd.DataFrame(X, columns=['description'])
        
        # Ensure description column exists and handle missing values
        if 'description' not in df_X.columns:
            df_X['description'] = ''
        df_X['description'] = df_X['description'].fillna('')
        
        # Tax-related keywords
        df_X['has_tax_keyword'] = df_X['description'].str.contains(
            'tckn|vkn|vergi|kesinti', case=False, regex=True, na=False
        ).astype(int)
        
        # Code pattern detection
        df_X['has_code_pattern'] = df_X['description'].apply(
            lambda x: 1 if re.search(r'\d', str(x)) and re.search(r'[a-zA-Z]', str(x)) and ' ' not in str(x) else 0
        )
        
        df_X['desc_length'] = df_X['description'].str.len().fillna(0)
        df_X['desc_digit_ratio'] = df_X['description'].apply(
            lambda x: sum(c.isdigit() for c in str(x)) / (len(str(x)) + 1e-6)
        )
        
        return df_X[['has_tax_keyword', 'has_code_pattern', 'desc_length', 'desc_digit_ratio']].values

# To collect all the data preparation responsibilities in one place:
# loading, validation, target generation, cleaning, split and preprocessor generation.
# This way we ensure the same transformations in the training and prediction phases.
class PreparationService:
    def __init__(self, data_path: str = "app/data/temp_data.csv") -> None:
        self.data_path = Path(data_path)

    def load_raw(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info("Loaded raw data from {path} with shape {shape}", path=str(self.data_path), shape=df.shape)
        return df

    def validate_columns(self, df: pd.DataFrame) -> None:
        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]
        if missing_features or missing_targets:
            raise ValueError(
                f"Missing columns. Features: {missing_features}, Targets: {missing_targets}"
            )

    # Create a single multiclass target based on which Y column is non-null
    def _build_target(self, df: pd.DataFrame) -> pd.Series:

        def pick_label(row: pd.Series) -> str | None:
            flags = {
                "seller": pd.notna(row.get("seller_number")),
                "customer": pd.notna(row.get("customer_number")),
                "account": pd.notna(row.get("main_account")),
            }
            on_count = sum(1 for v in flags.values() if v)
            if on_count == 1:
                for k, v in flags.items():
                    if v:
                        return k
            return None

        target = df.apply(pick_label, axis=1)
        return target

    def prepare_and_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.validate_columns(df)

        target = self._build_target(df)
        mask_valid = target.notna()
        df = df.loc[mask_valid].copy()
        target = target.loc[mask_valid].astype(str)

       
        df['description'] = df['description'].fillna("")
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

        X = df[FEATURE_COLUMNS].copy()
        y = target.to_frame(name="target_type")

        # 150/50 split for 200 rows; otherwise 25% test
        if len(df) >= 200:
            test_size = 50 / len(df)
        else:
            test_size = 0.25

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y["target_type"]
        )

        logger.info(
            "Split data: X_train={xt}, X_test={xv}, y_train={yt}, y_test={yv}",
            xt=X_train.shape,
            xv=X_test.shape,
            yt=y_train.shape,
            yv=y_test.shape,
        )
        return X_train, X_test, y_train, y_test

    def build_preprocessor(self, X: pd.DataFrame):
        """Build advanced preprocessor with multiple text features"""
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        
        categorical_cols = ['company_code', 'document_number', 'payment_type', 'currency_code', 'transaction_type']
        numeric_cols = ['amount']
        
        # Word-level TF-IDF
        word_tfidf = TfidfVectorizer(
            preprocessor=normalize_text, 
            analyzer='word', 
            ngram_range=(1, 2),
            min_df=1,  
            max_features=3000,  
            sublinear_tf=True
        )
        
        # Character-level TF-IDF
        char_tfidf = TfidfVectorizer(
            preprocessor=normalize_text, 
            analyzer='char_wb', 
            ngram_range=(3, 5),
            min_df=1,  
            max_features=2000,  
            sublinear_tf=True
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('desc_word', word_tfidf, 'description'),
                ('desc_char', char_tfidf, 'description'),
                ('custom_text_features', Pipeline([
                    ('extractor', CustomTextFeatures()),
                    ('scaler', StandardScaler())
                ]), 'description'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols),
            ],
            remainder='drop'
        )
        return preprocessor


