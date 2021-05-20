import gzip
import os
from typing import Dict, List
from urllib.request import urlretrieve

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
    # df_beta = df_vars["cov_xy"].divide(df_vars["var_x"])
    # df_beta.columns = pd.MultiIndex.from_tuples(
    #     [("beta_1", *c) for c in df_beta.columns],
    #     names=["stat", "win_type", "win_length"],
    # )
    # df_betas = pd.concat([df_vars, df_beta], axis=1)
    df_betas = df_betas.swaplevel("stat", "win_type", axis=1)
    df_betas = df_betas.sort_index(axis=1)

    return df_betas


def get_moments(df_betas: pd.DataFrame, start_date: str = "2016-02-01"):
    """
    Calculates moments for each series of betas and returns as
    a dataframe.
    """

    df_select = df_betas.swaplevel("stat", "win_type", axis=1)["beta_1"]
    records = []
    for s in df_select.items():
        moments = stats.describe(s[1].loc[start_date:].dropna())
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
    fig = px.bar(
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


IS_labels = [
    ("obs", lambda x: f"{x:>7d}"),
    ("min:max", lambda x: f"{x[0]:>0.4f}:{x[1]:>0.3f}"),
    ("mean", lambda x: f"{x:>7.4f}"),
    ("std", lambda x: f"{x:>7.4f}"),
    ("skewness", lambda x: f"{x:>7.4f}"),
    ("kurtosis", lambda x: f"{x:>7.4f}"),
]


def get_moments_annotation(
    s: pd.Series,
    xref: str,
    yref: str,
    x: float,
    y: float,
    xanchor: str,
    title: str,
    labels: List,
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """
    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    sharpe = s.mean() / s.std()

    return go.layout.Annotation(
        text=(
            f"<b>sharpe: {sharpe:>8.4f}</b><br>"
            + ("<br>").join(
                [f"{k[0]:<9}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=0.5,
        borderpad=2,
        bgcolor="white",
        xanchor=xanchor,
        yanchor="top",
    )


def make_components_chart(
    yc_L: str,
    fx_B: str,
    fx_L: str,
    libor: str,
    leverage: float,
    date_range: pd.date_range,
    dfs_yc: Dict,
    dfs_fx: Dict,
    dfs_libor: Dict,
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"5-Year Yield: {yc_L}",
            f"FX Rate: {fx_L}:{fx_B}",
            f"3 Month Libor: {libor}",
            f"FX Rate: {fx_B}:USD",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": True}],
        ],
    )

    # Lend market yield
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"],
            line=dict(width=1, color=COLORS[0]),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"].pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Borrow market fx
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate,
            line=dict(width=1, color=COLORS[0]),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=True,
    )

    # Borrow market funding cost
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_libor[libor].loc[date_range].value,
            line=dict(width=1, color=COLORS[0]),
            name=libor,
        ),
        row=2,
        col=1,
    )

    # Lend market fx cost
    # =================
    fx_BL = (
        dfs_fx[fx_L].loc[date_range].loc[date_range].rate
        / dfs_fx[fx_B].loc[date_range].rate
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL,
            line=dict(width=1, color=COLORS[0]),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="grey", mirror=True)
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="grey", mirror=True, tickformat="0.1f"
    )

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Underlying Securities: "
            f"{date_range.min().strftime('%Y-%m-%d')}"
            f" - {date_range.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        yaxis3=dict(tickformat="0.3f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    return fig


def make_returns_chart(df_ret: pd.DataFrame) -> go.Figure:

    fx_B, yc_L = df_ret.name.split(",")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Weekly Returns",
            f"Returns Distribution",
            f"Cumulative Returns",
            f"Q/Q Plot",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # Returns Distribution
    returns = pd.cut(df_ret.per_return, 50).value_counts().sort_index()
    midpoints = returns.index.map(lambda interval: interval.right).to_numpy()
    norm_dist = stats.norm.pdf(
        midpoints, loc=df_ret.per_return.mean(), scale=df_ret.per_return.std()
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return * 100,
            line=dict(width=1, color=COLORS[0]),
            name="return",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return.cumsum() * 100,
            line=dict(width=1, color=COLORS[0]),
            name="cum. return",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[interval.mid for interval in returns.index],
            y=returns / returns.sum() * 100,
            name="pct. of returns",
            marker=dict(color=COLORS[0]),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in returns.index],
            y=norm_dist / norm_dist.sum() * 100,
            name="normal",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=2,
    )

    # Q/Q Data
    returns_norm = (
        (df_ret.per_return - df_ret.per_return.mean()) / df_ret.per_return.std()
    ).sort_values()
    norm_dist = pd.Series(
        list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(df_ret.per_return)))),
        name="normal",
    )

    fig.append_trace(
        go.Scatter(
            x=norm_dist,
            y=returns_norm,
            name="return norm.",
            mode="markers",
            marker=dict(color=COLORS[0], size=3),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=norm_dist,
            y=norm_dist,
            name="norm.",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=2,
        col=2,
    )

    fig.add_annotation(
        text=(f"{df_ret.per_return.cumsum()[-1] * 100:0.2f}"),
        xref="paper",
        yref="y3",
        x=0.465,
        y=df_ret.per_return.cumsum()[-1] * 100,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        get_moments_annotation(
            df_ret.per_return,
            xref="paper",
            yref="paper",
            x=0.81,
            y=0.23,
            xanchor="left",
            title="Returns",
            labels=IS_labels,
        ),
        font=dict(size=6, family="Courier New, monospace"),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Returns: "
            f"{df_ret.index.min().strftime('%Y-%m-%d')}"
            f" - {df_ret.index.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=100),
        yaxis=dict(tickformat="0.1f"),
        yaxis3=dict(tickformat="0.1f"),
        yaxis2=dict(tickformat="0.1f"),
        yaxis4=dict(tickformat="0.1f"),
        xaxis2=dict(tickformat="0.1f"),
        xaxis4=dict(tickformat="0.1f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    fig.update_annotations(font=dict(size=10))

    return fig
