import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from itertools import islice

st.set_page_config(layout='wide')
st.html('styles.html')

@st.cache_data
def get_data():
    ticker_df = pd.read_excel('data/Financial Data.xlsx', sheet_name='ticker')

    history_df = {}
    for ticker in list(ticker_df['Ticker']):
        d = pd.read_excel('data/Financial Data.xlsx', sheet_name=ticker)
        history_df[ticker] = d
    
    return ticker_df, history_df


def transform_data(ticker_df, history_df):
    ticker_df['Last Trade time'] = pd.to_datetime(
        ticker_df['Last Trade time'], 
        dayfirst=True
    )

    for col in ["Last Price", "Previous Day Price", "Change", "Change Pct", "Volume", "Volume Avg", "Shares", "Day High", "Day Low", "Market Cap", "P/E Ratio", "EPS"]:
        ticker_df[col] =  pd.to_numeric(
            ticker_df[col],
            'coerce'
        )

    for ticker in list(ticker_df['Ticker']):
        history_df[ticker]['Date'] = pd.to_datetime(
            history_df[ticker]['Date'], 
            dayfirst=True
        )
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            history_df[ticker][col] =  pd.to_numeric(
                history_df[ticker][col],
                'coerce'
            )
    ticker_to_open =  [
        list(history_df[t]['Open'])
        for t in list(ticker_df['Ticker'])
    ]

    ticker_df['Open'] = ticker_to_open
    
    return ticker_df, history_df


def display_overview(ticker_df):
    ### Style for the table
    def format_currency(val):
        return "$ {:,.2f}".format(val)

    def format_percentage(val):
        return "{:,.2f} %".format(val)
    
    def format_number(val):
        return "{:,.2f}".format(val)
    
    def format_datetime(val):
        return val.strftime("%Y-%m-%d %H:%M:%S")

    def apply_ood_row_class(row):
        return [
            'background-color: #f4f4f4' if row.name % 2 != 0
            else "background-color: #ffffff" for _ in row
        ]

    def format_change(val):
        return "color: red" if (val < 0) else "color: green"

    ### Applying all styles
    with st.expander("📊 Stocks Preview"):
        styled_df = ticker_df.style.format(
            {
                "Last Price": format_currency,
                "Change Pct": format_percentage,
                "Previous Date Price": format_currency,
                "Change": format_currency,
                "Volume": format_number,
                "Volume Avg": format_number,
                "Share": format_number,
                "Day High": format_currency,
                "Day Low": format_currency,
                "Market Cap": format_currency,
                "P/E Ratio": format_number,
                "EPS": format_number,
                "Last Trade time": format_datetime
            }
        ).apply(apply_ood_row_class, axis=1).map(format_change, subset=["Change Pct", "Change"])

        st.dataframe(
            styled_df, 
            column_config={
                'Open': st.column_config.AreaChartColumn(
                    'Last 12 Month',
                    width='large',
                    help='Open Price for the last 12 Months'
                )
            },
            hide_index=True,
            height=250,
            use_container_width=True
        )


def plot_candlestick(history_df):
    f_candle = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
    )

    f_candle.add_trace(
        go.Candlestick(
            x=history_df.index,
            open=history_df['Open'],
            high=history_df['High'],
            low=history_df['Low'],
            close=history_df['Close'],
            name='Dollars'
        ),
        row=1,
        col=1,
    )

    f_candle.add_trace(
        go.Bar(
            x=history_df.index, 
            y=history_df["Volume"], 
            name="Volume Traded"
        ),
        row=2,
        col=1,
    )

    f_candle.update_layout(
        title="Stock Price Trends",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        yaxis1=dict(title="OHLC"),
        yaxis2=dict(title="Volume"),
        hovermode="x",
    )

    f_candle.update_layout(
        title_font_family="Arial",
        title_font_color="#174C4F",
        title_font_size=28,
        font_size=16,
        margin=dict(l=80, r=80, t=100, b=80, pad=0),
        height=500,
    )
        
    f_candle.update_layout(
        legend=dict(
            orientation="h",  # Alineación horizontal
            yanchor="bottom",  # Ancla en la parte inferior de la leyenda
            y=1,  # Coloca la leyenda ligeramente por encima del gráfico
            xanchor="center",  # Ancla la leyenda en el centro horizontalmente
            x=0.5,  # Centra la leyenda horizontalmente
        )
    )

    f_candle.update_xaxes(title_text="Date", row=2, col=1)
    f_candle.update_traces(selector=dict(name="Dollars"), showlegend=True)

    return f_candle

@st.fragment
def display_simbol_history(ticker_df, history_df):
    left_widget, right_widget, _ = st.columns([1,1,1.5])

    selected_ticker = left_widget.selectbox(
        "📈 Currently Showing",
        list(history_df.keys())
    )

    selected_period = right_widget.selectbox(
        "📅 Period",
        ("Week","Month", "Trimester", "Year"),
        1
    )

    history_df = history_df[selected_ticker]

    history_df["Date"] = pd.to_datetime(history_df["Date"], dayfirst=True)
    history_df = history_df.set_index("Date")
    mapping_period = {
        "Week":7,
        "Month":31,
        "Trimester":90,
        "Year":365
    }

    latest_date = history_df.index.max()
    date_filter = pd.to_datetime(latest_date - pd.Timedelta(mapping_period[selected_period], unit="d"))
    history_df = history_df[(history_df.index >= date_filter) & (history_df.index <= latest_date)]

    f_candle = plot_candlestick(history_df)

    left_chart, rigth_indicator =  st.columns([2,1.5])

    with left_chart:
        st.html('<span class="column_plotly"></span>')
        st.plotly_chart(f_candle, use_container_width=True)

    with rigth_indicator:
        st.html('<span class="column_indicator"></span>')
        st.subheader("Period Metrics")
        l, r = st.columns(2)

        with l:
            st.html('<span class="low_indicator"></span>')
            st.metric(
                "Lowest Volume Day Trade",
                f'{history_df["Volume"].min():,}'
            )
            st.metric(
                "Lowest Close Price",
                f'${history_df["Close"].min():,}'            
            )

        with r:
            st.html('<span class="high_indicator"></span>')
            st.metric(
                "Highest Volume Day Trade",
                f'{history_df["Volume"].max():,}'
            )
            st.metric(
                "Highest Close Price",
                f'${history_df["Close"].max():,}'            
            )
        with st.container():
            st.html('<span class="bottom_indicator"></span>')
            st.metric(
                "Average Daily Volumne",
                f'{int(history_df["Volume"].mean()):,}'
            )
            st.metric(
                "Current Market Cap",
                "${:,}".format(
                    ticker_df[ticker_df["Ticker"] == selected_ticker]["Market Cap"].values[0]
                )            
            )



def batched(iterable, n_cols):
    if n_cols < 1:
        raise ValueError('n muest be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n_cols)):
        yield batch

def plot_sparkline(data):
    fig_spark = go.Figure(
        data=go.Scatter(
            y=data,
            mode="lines",
            fill="tozeroy",
            line_color="red",
            fillcolor="pink",
        ),
    )
    fig_spark.update_traces(hovertemplate="Price: $ %{y:.2f}")
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=False, fixedrange=True)
    fig_spark.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=50,
        margin=dict(t=0, l=0, b=0, r=0, pad=0),
        autosize =True,
        width=None,
    )
    return fig_spark

def display_watchlist_card(ticker, symbol_name, last_price, change_pct, open):
    with st.container(border=True):
        st.html(f'<span class="watchlist_card"></span>')

        tl, tr = st.columns([3,1])
        bl, br = st.columns([1,1])

        with tl:
            st.html(f'<span class="watchlist_symbol_name"></span>')
            st.markdown(f'{symbol_name}')

        with tr:
            st.html(f'<span class="watchlist_ticker"></span>')
            st.markdown(f'{ticker}')


        with bl:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f'Current Value')

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f'$ {last_price}')
        with br:
            negative_gradient =  float(change_pct) < 0
            st.html(f'<span class="watchlist_change"></span>')
            st.markdown(
                f":{'red' if negative_gradient else 'green'}[{'▼' if negative_gradient else '▲'} {change_pct} %]"
            )


        with st.container():
            fig_spark = plot_sparkline(open)
            st.html(f'<span class="watchlist_br"></span>')
            st.plotly_chart(
                fig_spark,
                config=dict(displayModeBar=False),
                use_container_width=True
            )

def display_watchlist(ticker_df):
    n_cols =  4
    ticker_df2 = ticker_df.copy()
    ticker_df2.columns = ticker_df.columns.str.replace(' ', '_')
    for row in batched(ticker_df2.itertuples(), n_cols):
        cols = st.columns(n_cols)
        for col, ticker, in zip(cols,row):
            if ticker:
                with col:
                    display_watchlist_card(
                        ticker.Ticker,
                        ticker.Symbol_Name,
                        ticker.Last_Price,
                        ticker.Change_Pct,
                        ticker.Open,
                    )


st.html('<h1 class="title">Stocks Dashboard</h1>')

ticker_df, history_df = get_data()
ticker_df, history_df = transform_data(ticker_df, history_df)

display_watchlist(ticker_df)

st.divider()

display_simbol_history(ticker_df, history_df)
display_overview(ticker_df)


