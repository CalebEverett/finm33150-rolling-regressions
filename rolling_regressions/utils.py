import gzip
import os
from typing import Dict, List
from urllib.request import urlretrieve
import warnings

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from canvasapi import Canvas
import numpy as np
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from plotly.subplots import make_subplots
import quandl
import requests
from scipy import stats
from tqdm.notebook import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

# =============================================================================
# Credentials
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Canvas
# =============================================================================


def download_files(filename_frag: str):
    """Downloads files from Canvas with `filename_frag` in filename."""

    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    for f in course.get_files():
        if filename_frag in f.filename:
            print(f.filename, f.id)
            file = course.get_file(f.id)
            file.download(file.filename)


# =============================================================================
# Reading Data
# =============================================================================


def get_trade_data(pair: str, year: str, path: str = "accumulation_opportunity/data"):
    """Reads local gzipped trade data file and return dataframe."""

    dtypes = {
        "PriceMillionths": int,
        "Side": int,
        "SizeBillionths": int,
        "timestamp_utc_nanoseconds": int,
    }

    filename = f"trades_narrow_{pair}_{year}.delim.gz"
    delimiter = {"2018": "|", "2021": "\t"}[year]

    with gzip.open(f"{path}/{filename}") as f:
        df = pd.read_csv(f, delimiter=delimiter, usecols=dtypes.keys(), dtype=dtypes)

    df.timestamp_utc_nanoseconds = pd.to_datetime(df.timestamp_utc_nanoseconds)

    return df.set_index("timestamp_utc_nanoseconds")


# =============================================================================
# Price Data
# =============================================================================


def get_table(dataset_code: str, database_code: str = "ZACKS"):
    """Downloads Zacks fundamental table from export api to local zip file."""

    url = (
        f"https://www.quandl.com/api/v3/datatables/{database_code}/{dataset_code}.json"
    )
    r = requests.get(
        url, params={"api_key": os.getenv("QUANDL_API_KEY"), "qopts.export": "true"}
    )
    data = r.json()
    urlretrieve(
        data["datatable_bulk_download"]["file"]["link"],
        f"zacks_{dataset_code.lower()}.zip",
    )


def load_table_files(table_filenames: Dict):
    """Loads Zacks fundamentals tables from csv files."""

    dfs = []
    for v in tqdm(table_filenames.values()):
        dfs.append(pd.read_csv(v, low_memory=False))

    return dfs


def get_hash(string: str) -> str:
    """Returns md5 hash of string."""

    return hashlib.md5(str(string).encode()).hexdigest()


def fetch_ticker(
    dataset_code: str, query_params: Dict = None, database_code: str = "EOD"
):
    """Fetches price data for a single ticker."""

    url = f"https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}.json"

    params = dict(api_key=os.getenv("QUANDL_API_KEY"))
    if query_params is not None:
        params = dict(**params, **query_params)

    r = requests.get(url, params=params)

    dataset = r.json()["dataset"]
    df = pd.DataFrame(
        dataset["data"], columns=[c.lower() for c in dataset["column_names"]]
    )
    df["ticker"] = dataset["dataset_code"]

    return df.sort_values("date")


def fetch_all_tickers(tickers: List, query_params: Dict) -> pd.DataFrame:
    """Fetches price data from Quandl for each ticker in provide list and
    returns a dataframe of them concatenated together.
    """

    df_prices = pd.DataFrame()
    for t in tqdm(tickers):
        try:
            df = fetch_ticker(t, query_params)
            df_prices = pd.concat([df_prices, df])
        except:
            print(f"Couldn't get prices for {t}.")

    not_missing_data = (
        df_prices.set_index(["ticker", "date"])[["adj_close"]]
        .unstack("date")
        .isna()
        .sum(axis=1)
        == 0
    )

    df_prices = df_prices[
        df_prices.ticker.isin(not_missing_data[not_missing_data].index)
    ]

    return df_prices.set_index(["ticker", "date"])


# =============================================================================
# Download Preprocessed Files from S3
# =============================================================================


def upload_s3_file(filename: str):
    """Uploads file to S3. Requires credentials with write permissions to exist
    as environment variables.
    """

    client = boto3.client("s3")
    client.upload_file(filename, "finm33150", filename)


def download_s3_file(filename: str):
    """Downloads file from read only S3 bucket."""

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    client.download_file("finm33150", filename, filename)


# =============================================================================
# Data Preparation
# =============================================================================


def get_betas(
    df_ret: pd.DataFrame, win_lengths: List, nobs_oos: int = 6
) -> pd.DataFrame:
    """
    Takes dataframe of returns by date by ticker and returns dataframe of coefficients
    calculated from exponentially weighted moving average, boxcar and forward boxcar
    windows. EWN averages are based on alphas as 1/w. Boxcar windows are based on 2 * w.
    Forward boxcar windows are for the next `nobs_oos` observations and the same value is
    returned under each time heading to facilitate comparisons.
    """

    col_list = []
    for w in tqdm(win_lengths):
        for s in df_ret.columns[:-1]:
            df_pair = df_ret.loc[:, [s, "SPY"]]
            for win_type, df_cov in {
                "exp_wm": df_pair.ewm(alpha=1 / w).cov(),
                "boxcar": df_pair.rolling(window=2 * w).cov(),
                "boxcar_fwd": df_pair.rolling(window=nobs_oos)
                .cov()
                .groupby("ticker")
                .shift(-nobs_oos),
            }.items():
                s_var = df_cov[s].xs(s, level=1)
                s_var.name = ("var_x", win_type, f"t_{w:02d}", s)

                s_cov = df_cov[s].xs("SPY", level=1)
                s_cov.name = ("cov_xy", win_type, f"t_{w:02d}", s)

                s_beta = s_cov / s_var
                s_beta.name = ("beta_1", win_type, f"t_{w:02d}", s)

                col_list.extend([s_beta, s_var, s_cov])

    df_betas = pd.concat(col_list, axis=1)
    df_betas.columns.names = ["stat", "win_type", "win_length", "ticker"]
    df_betas = df_betas.stack("ticker")
    df_betas = df_betas.swaplevel("stat", "win_type", axis=1)
    df_betas = df_betas.sort_index(axis=1)

    return df_betas


def get_moments(df_betas: pd.DataFrame):
    """
    Calculates moments for each series of betas and returns as
    a dataframe.
    """

    df_select = df_betas.swaplevel("stat", "win_type", axis=1)["beta_1"]
    records = []
    for s in df_select.items():
        moments = stats.describe(s[1].dropna())
        for stat in ["mean", "variance", "skewness", "kurtosis"]:
            record = {"win_type": s[0][0], "win_length": s[0][1]}
            record["stat"] = stat
            record["stat_value"] = moments._asdict()[stat]
            records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.D3


def make_moments_chart(df_moments: pd.DataFrame) -> go.Figure:
    """
    Returns figure for moments comparison chart.
    """
    fig = px.line(
        df_moments,
        x="win_length",
        y="stat_value",
        facet_col="stat",
        facet_row="win_type",
        title="Comparison of Moments",
        height=600,
    )
    for col in [1, 2, 3, 4]:
        fig.update_yaxes(matches=f"y{col}", col=col, showticklabels=True)

    return fig


def make_return_chart(df_ret: pd.DataFrame) -> go.Figure:
    """
    Returns average return chart figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.loc[:, df_ret.columns != "SPY"].cumsum(axis=0).mean(axis=1),
            name="portfolio mean",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret["SPY"].cumsum(),
            name="SPY",
        )
    )

    fig.update_layout(title="Cumulative Return")

    return fig


def show_histograms(df_betas: pd.DataFrame) -> go.Figure:

    for win_length in df_betas.columns.levels[-1]:
        df_select = df_betas[
            [
                ("exp_wm", "beta_1", win_length),
                ("boxcar", "beta_1", win_length),
                ("boxcar_fwd", "beta_1", win_length),
            ]
        ].dropna()

        fig = px.histogram(
            df_select.stack(["win_type", "stat"]).reset_index(),
            x=win_length,
            color="win_type",
            barmode="overlay",
            title=f"Distribution of Coefficients: {win_length}",
            histnorm="percent",
            marginal="box",
            opacity=0.7,
        )
        fig.update_xaxes(range=(-2, 2))

        fig.write_image(f"images/distribution_{win_length}.png", height=600, width=1200)

        fig.show(renderer="png", height=600, width=1200)
