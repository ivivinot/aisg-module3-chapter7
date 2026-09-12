import os
import requests
import pandas as pd
from sqlalchemy import create_engine, inspect


class DataDownloader:
    """Load data from CSV, JSON, Excel (.xls/.xlsx) or SQLite (.db).

    Optionally download the file from a URL first.
    """

    def __init__(self, filepath: str = None, table_name: str = None, data_folder: str = "data"):
        self.filepath = filepath
        self.data_folder = data_folder
        self.table_name = table_name
        self.extension = os.path.splitext(filepath)[1].lower() if filepath else None
        if data_folder:
            os.makedirs(data_folder, exist_ok=True)

    def download_url(self, url: str, filename: str) -> str:
        """Download a file from a URL; skip if it is already on disk."""
        filepath = os.path.join(self.data_folder, filename)
        if os.path.exists(filepath):
            print(f"[ok] File already exists: {filepath}")
            self._update_path(filepath)
            return filepath

        print(f"[..] Downloading from {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"[ok] Saved to: {filepath}")
        self._update_path(filepath)
        return filepath

    def data_load(self) -> pd.DataFrame:
        """Load the data based on the file extension."""
        ext = self.extension
        if ext == ".csv":
            df = pd.read_csv(self.filepath)
        elif ext == ".json":
            df = pd.read_json(self.filepath)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(self.filepath, engine="openpyxl")
        elif ext == ".db":
            df = self._load_db()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(f"[ok] Loaded {len(df):,} rows x {df.shape[1]} columns from {self.filepath}")
        return df

    def _load_db(self) -> pd.DataFrame:
        engine = create_engine(f"sqlite:///{self.filepath}")
        with engine.connect() as conn:
            tables = inspect(engine).get_table_names()
            if not tables:
                raise RuntimeError("No tables found in database.")
            table = self.table_name or tables[0]
            print(f"[..] Loading table: {table}")
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    def _update_path(self, filepath: str) -> None:
        self.filepath = filepath
        self.extension = os.path.splitext(filepath)[1].lower()
