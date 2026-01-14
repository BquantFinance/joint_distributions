"""
Joint Distribution Evolution Visualizer
@Gsnchez | bquantfinance.com

Visualize how the joint distribution of asset returns evolves over time.
Enhanced with tail dependence, regime detection, and conditional analysis.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from itertools import combinations
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Joint Distribution Evolution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0f1117;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f2f6;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #9ca3af;
        font-weight: 400;
    }
    
    .brand-link {
        color: #c9a962;
        text-decoration: none;
        font-weight: 500;
    }
    
    .brand-link:hover {
        color: #e0c478;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f0f2f6;
    }
    
    .info-box {
        background: rgba(255,255,255,0.02);
        border-left: 2px solid #c9a962;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #9ca3af;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.03);
        border-color: rgba(255,255,255,0.08);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0a0c10;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    
    .sidebar-header {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    footer {
        text-align: center;
        padding: 2rem 0;
        color: #4b5563;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2rem;
    }
    
    .stButton > button {
        background: #c9a962;
        color: #0f1117;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #d4b872;
        box-shadow: 0 4px 12px rgba(201, 169, 98, 0.25);
    }
    
    .tail-warning {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .tail-ok {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <div class="main-title">Joint Distribution Evolution</div>
    <div class="main-subtitle">
        <a href="https://twitter.com/Gsnchez" class="brand-link" target="_blank">@Gsnchez</a> · 
        <a href="https://bquantfinance.com" class="brand-link" target="_blank">bquantfinance.com</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & PRESETS
# ============================================================================
PRESET_PAIRS = {
    "Custom": [],
    "Risk-On / Risk-Off (SPY + TLT)": ["SPY", "TLT"],
    "Stocks + Gold Hedge (SPY + GLD)": ["SPY", "GLD"],
    "Tech vs Value (QQQ + VTV)": ["QQQ", "VTV"],
    "US vs International (SPY + EFA)": ["SPY", "EFA"],
    "Stocks + Bonds + Gold (SPY + TLT + GLD)": ["SPY", "TLT", "GLD"],
    "Growth vs Defensives (QQQ + XLU)": ["QQQ", "XLU"],
    "Large vs Small Cap (SPY + IWM)": ["SPY", "IWM"],
}

CRISIS_DATES = {
    "2008 Financial Crisis": ("2008-09-15", "2009-03-09"),
    "2010 Flash Crash": ("2010-05-06", "2010-05-07"),
    "2011 Debt Ceiling": ("2011-08-01", "2011-08-15"),
    "2015 China Slowdown": ("2015-08-18", "2015-08-26"),
    "2018 Vol Spike": ("2018-02-02", "2018-02-12"),
    "2020 COVID Crash": ("2020-02-20", "2020-03-23"),
    "2022 Rate Shock": ("2022-01-03", "2022-10-12"),
}


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================
def compute_tail_dependence(x: np.ndarray, y: np.ndarray, u: float = 0.05) -> float:
    """
    Compute lower tail dependence coefficient lambda_L.
    Estimates P(Y < F_Y^{-1}(u) | X < F_X^{-1}(u))
    """
    n = len(x)
    if n < 20:
        return np.nan
    
    # Convert to uniform margins using empirical CDF
    x_ranks = stats.rankdata(x) / (n + 1)
    y_ranks = stats.rankdata(y) / (n + 1)
    
    # Count joint tail exceedances
    x_in_tail = x_ranks <= u
    y_in_tail = y_ranks <= u
    
    n_x_tail = x_in_tail.sum()
    if n_x_tail == 0:
        return np.nan
    
    joint_tail = (x_in_tail & y_in_tail).sum()
    
    return joint_tail / n_x_tail


def compute_rolling_correlation(returns: pd.DataFrame, pair: tuple, window: int) -> pd.Series:
    """Compute rolling correlation for a pair."""
    return returns[pair[0]].rolling(window).corr(returns[pair[1]])


def compute_rolling_tail_dependence(returns: pd.DataFrame, pair: tuple, window: int, u: float = 0.05) -> pd.Series:
    """Compute rolling tail dependence coefficient."""
    result = []
    x = returns[pair[0]].values
    y = returns[pair[1]].values
    
    for i in range(len(returns)):
        if i < window:
            result.append(np.nan)
        else:
            x_win = x[i-window:i]
            y_win = y[i-window:i]
            result.append(compute_tail_dependence(x_win, y_win, u))
    
    return pd.Series(result, index=returns.index)


def detect_regime_changes(corr_series: pd.Series, threshold: float = 1.5) -> list:
    """Detect significant correlation regime changes."""
    if corr_series.isna().all():
        return []
    
    # Compute rolling mean and std
    rolling_mean = corr_series.rolling(60, min_periods=30).mean()
    rolling_std = corr_series.rolling(60, min_periods=30).std()
    
    # Z-score
    z_scores = (corr_series - rolling_mean) / rolling_std
    
    # Find significant deviations
    regimes = []
    in_regime = False
    regime_start = None
    regime_type = None
    
    for i, (date, z) in enumerate(z_scores.items()):
        if pd.isna(z):
            continue
            
        if not in_regime and abs(z) > threshold:
            in_regime = True
            regime_start = date
            regime_type = "spike" if z > 0 else "crash"
        elif in_regime and abs(z) < threshold * 0.5:
            if regime_start is not None:
                regimes.append({
                    'start': regime_start,
                    'end': date,
                    'type': regime_type,
                    'peak_z': z_scores.loc[regime_start:date].abs().max()
                })
            in_regime = False
            regime_start = None
    
    return regimes


def compute_conditional_distribution(returns: pd.DataFrame, pair: tuple, 
                                      condition_asset: str, condition_value: float,
                                      window_size: int, tolerance: float = 0.5) -> tuple:
    """
    Compute conditional distribution: P(Y | X approx condition_value)
    Uses kernel density estimation with data points near the condition.
    """
    window_data = returns.iloc[-window_size:]
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    
    if condition_asset == pair[0]:
        condition_data = x
        response_data = y
    else:
        condition_data = y
        response_data = x
    
    # Find points near the condition
    weights = np.exp(-0.5 * ((condition_data - condition_value) / tolerance) ** 2)
    weights = weights / weights.sum()
    
    # Return weighted samples for histogram
    return response_data, weights


def generate_bivariate_normal(x_range: tuple, y_range: tuple, 
                               mean_x: float, mean_y: float,
                               std_x: float, std_y: float, 
                               corr: float, grid_size: int = 80) -> tuple:
    """Generate theoretical bivariate normal surface."""
    xx, yy = np.mgrid[x_range[0]:x_range[1]:complex(grid_size), 
                      y_range[0]:y_range[1]:complex(grid_size)]
    
    # Bivariate normal PDF
    pos = np.dstack((xx, yy))
    mean = [mean_x, mean_y]
    cov = [[std_x**2, corr * std_x * std_y],
           [corr * std_x * std_y, std_y**2]]
    
    rv = stats.multivariate_normal(mean, cov)
    density = rv.pdf(pos)
    
    # Normalize
    density_norm = (density - density.min()) / (density.max() - density.min())
    
    return xx, yy, density_norm


# ============================================================================
# PLOTLY VISUALIZATION FUNCTIONS
# ============================================================================
def create_plotly_3d_single(
    returns: pd.DataFrame,
    pair: tuple,
    window_size: int,
    date_str: str,
    show_gaussian: bool = False
) -> go.Figure:
    """Create spectacular interactive 3D Plotly surface for a single pair."""
    
    window_data = returns.iloc[-window_size:]
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    corr = np.corrcoef(x, y)[0, 1]
    tail_dep = compute_tail_dependence(x / 100, y / 100, u=0.05)
    
    # KDE
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    x_range = xmax - xmin
    y_range = ymax - ymin
    xmin -= x_range * 0.3
    xmax += x_range * 0.3
    ymin -= y_range * 0.3
    ymax += y_range * 0.3
    
    grid_size = 100
    xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    try:
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        density = np.reshape(kernel(positions).T, xx.shape)
        density = gaussian_filter(density, sigma=1.2)
        density_norm = (density - density.min()) / (density.max() - density.min())
    except:
        density_norm = np.zeros_like(xx)
    
    # Spectacular custom colorscale
    colorscale = [
        [0.0, 'rgba(15, 17, 23, 0.0)'],
        [0.05, 'rgba(20, 30, 50, 0.3)'],
        [0.15, 'rgba(30, 58, 95, 0.6)'],
        [0.3, 'rgba(45, 97, 135, 0.8)'],
        [0.45, 'rgba(74, 158, 187, 0.9)'],
        [0.6, 'rgba(126, 204, 232, 0.95)'],
        [0.75, 'rgba(184, 224, 240, 0.98)'],
        [0.88, 'rgba(232, 213, 163, 1.0)'],
        [1.0, 'rgba(201, 169, 98, 1.0)'],
    ]
    
    # Gaussian overlay colorscale (red/orange)
    gaussian_colorscale = [
        [0.0, 'rgba(50, 20, 20, 0.0)'],
        [0.3, 'rgba(120, 40, 40, 0.3)'],
        [0.6, 'rgba(180, 80, 60, 0.5)'],
        [1.0, 'rgba(239, 68, 68, 0.6)'],
    ]
    
    fig = go.Figure()
    
    # Main empirical surface
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=density_norm,
        colorscale=colorscale,
        showscale=False,
        opacity=0.97,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4, roughness=0.3, fresnel=0.3),
        lightposition=dict(x=0, y=0, z=2),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=2)),
        hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<br>Density: %{{z:.3f}}<extra></extra>',
        name='Empirical'
    ))
    
    # Gaussian overlay if enabled
    if show_gaussian:
        _, _, gaussian_density = generate_bivariate_normal(
            (xmin, xmax), (ymin, ymax),
            np.mean(x), np.mean(y),
            np.std(x), np.std(y),
            corr, grid_size
        )
        
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=gaussian_density * 0.95,  # Slightly lower to see both
            colorscale=gaussian_colorscale,
            showscale=False,
            opacity=0.5,
            lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2),
            contours=dict(z=dict(show=True, color='#ef4444', width=1)),
            hovertemplate='Gaussian<br>Density: %{z:.3f}<extra></extra>',
            name='Gaussian'
        ))
    
    # Scatter points on surface
    try:
        interp = RegularGridInterpolator(
            (np.linspace(xmin, xmax, grid_size), np.linspace(ymin, ymax, grid_size)),
            density_norm, method='linear', bounds_error=False, fill_value=0
        )
        z_scatter = interp(np.column_stack([x, y]))
        z_scatter = np.clip(z_scatter, 0, None) + 0.02
    except:
        z_scatter = np.full_like(x, 0.05)
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z_scatter,
        mode='markers',
        marker=dict(size=4, color='#c9a962', opacity=0.9, line=dict(color='#ffffff', width=0.5)),
        hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<extra></extra>',
        name='Returns'
    ))
    
    # Title with correlation AND tail dependence
    corr_color = '#ef4444' if corr > 0.5 else '#22c55e' if corr < -0.2 else '#c9a962'
    tail_color = '#ef4444' if (not np.isnan(tail_dep) and tail_dep > 0.3) else '#c9a962'
    tail_str = f'λ<sub>L</sub>={tail_dep:.2f}' if not np.isnan(tail_dep) else 'λ<sub>L</sub>=N/A'
    
    title_text = f'<b>{pair[0]} vs {pair[1]}</b>  ·  {date_str}  ·  <span style="color:{corr_color}">ρ={corr:.2f}</span>  ·  <span style="color:{tail_color}">{tail_str}</span>'
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=15, color='#f0f2f6')),
        scene=dict(
            xaxis=dict(title=f'{pair[0]} (%)', title_font=dict(color='#9ca3af', size=12),
                      tickfont=dict(color='#6b7280', size=10), gridcolor='rgba(75, 85, 99, 0.15)',
                      showbackground=False, showgrid=True, zeroline=False, showspikes=False),
            yaxis=dict(title=f'{pair[1]} (%)', title_font=dict(color='#9ca3af', size=12),
                      tickfont=dict(color='#6b7280', size=10), gridcolor='rgba(75, 85, 99, 0.15)',
                      showbackground=False, showgrid=True, zeroline=False, showspikes=False),
            zaxis=dict(title='', showticklabels=False, showgrid=False, showbackground=False,
                      zeroline=False, showspikes=False),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#0f1117', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0), showlegend=False, height=500
    )
    
    return fig


def create_plotly_3d_grid(
    returns: pd.DataFrame,
    pairs: list,
    window_size: int,
    date_str: str,
    show_gaussian: bool = False
) -> go.Figure:
    """Create spectacular interactive 3D Plotly surfaces for multiple pairs."""
    
    n_pairs = len(pairs)
    rows, cols = (1, n_pairs) if n_pairs <= 3 else (2, 3)
    
    specs = [[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = []
    
    window_data = returns.iloc[-window_size:]
    
    # Pre-compute stats for titles
    for pair in pairs:
        x = window_data[pair[0]].values * 100
        y = window_data[pair[1]].values * 100
        corr = np.corrcoef(x, y)[0, 1]
        tail_dep = compute_tail_dependence(x / 100, y / 100, u=0.05)
        tail_str = f'λ={tail_dep:.2f}' if not np.isnan(tail_dep) else ''
        subplot_titles.append(f'{pair[0]} vs {pair[1]} · ρ={corr:.2f} {tail_str}')
    
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles,
                        horizontal_spacing=0.02, vertical_spacing=0.08)
    
    colorscale = [
        [0.0, 'rgba(15, 17, 23, 0.0)'], [0.05, 'rgba(20, 30, 50, 0.3)'],
        [0.15, 'rgba(30, 58, 95, 0.6)'], [0.3, 'rgba(45, 97, 135, 0.8)'],
        [0.45, 'rgba(74, 158, 187, 0.9)'], [0.6, 'rgba(126, 204, 232, 0.95)'],
        [0.75, 'rgba(184, 224, 240, 0.98)'], [0.88, 'rgba(232, 213, 163, 1.0)'],
        [1.0, 'rgba(201, 169, 98, 1.0)'],
    ]
    
    for idx, pair in enumerate(pairs):
        row, col = idx // cols + 1, idx % cols + 1
        
        x = window_data[pair[0]].values * 100
        y = window_data[pair[1]].values * 100
        corr = np.corrcoef(x, y)[0, 1]
        
        try:
            xmin, xmax = np.percentile(x, [2, 98])
            ymin, ymax = np.percentile(y, [2, 98])
            xmin -= (xmax - xmin) * 0.3
            xmax += (xmax - xmin) * 0.3
            ymin -= (ymax - ymin) * 0.3
            ymax += (ymax - ymin) * 0.3
            
            grid_size = 70
            xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            density = gaussian_filter(density, sigma=1.0)
            density_norm = (density - density.min()) / (density.max() - density.min())
        except:
            grid_size = 50
            xx, yy = np.mgrid[-3:3:complex(grid_size), -3:3:complex(grid_size)]
            density_norm = np.zeros_like(xx)
            xmin, xmax, ymin, ymax = -3, 3, -3, 3
        
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=density_norm, colorscale=colorscale, showscale=False, opacity=0.97,
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4, roughness=0.3, fresnel=0.3),
            lightposition=dict(x=0, y=0, z=2),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
            hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<extra></extra>'
        ), row=row, col=col)
        
        # Scatter
        try:
            interp = RegularGridInterpolator(
                (np.linspace(xmin, xmax, grid_size), np.linspace(ymin, ymax, grid_size)),
                density_norm, method='linear', bounds_error=False, fill_value=0
            )
            z_scatter = np.clip(interp(np.column_stack([x, y])), 0, None) + 0.02
        except:
            z_scatter = np.full_like(x, 0.05)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z_scatter, mode='markers',
            marker=dict(size=3, color='#c9a962', opacity=0.85, line=dict(color='#ffffff', width=0.3)),
            hoverinfo='skip'
        ), row=row, col=col)
        
        scene_name = f'scene{idx + 1}' if idx > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(title=f'{pair[0]} (%)', title_font=dict(color='#9ca3af', size=10),
                          tickfont=dict(color='#6b7280', size=8), gridcolor='rgba(75, 85, 99, 0.12)',
                          showbackground=False, showgrid=True, zeroline=False, showspikes=False),
                yaxis=dict(title=f'{pair[1]} (%)', title_font=dict(color='#9ca3af', size=10),
                          tickfont=dict(color='#6b7280', size=8), gridcolor='rgba(75, 85, 99, 0.12)',
                          showbackground=False, showgrid=True, zeroline=False, showspikes=False),
                zaxis=dict(title='', showticklabels=False, showgrid=False, showbackground=False,
                          zeroline=False, showspikes=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.45), bgcolor='rgba(0,0,0,0)'
            )
        })
    
    fig.update_layout(
        title=dict(text=f'<b>Joint Distributions</b>  ·  {date_str}', x=0.5, font=dict(size=18, color='#f0f2f6')),
        paper_bgcolor='#0f1117', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=80, b=0), showlegend=False,
        height=450 if n_pairs <= 3 else 700
    )
    
    for annotation in fig.layout.annotations:
        annotation.font.color = '#f0f2f6'
        annotation.font.size = 11
    
    return fig


def create_correlation_timeline(
    returns: pd.DataFrame,
    pairs: list,
    window_size: int,
    crisis_dates: dict = None,
    show_tail_dep: bool = True
) -> go.Figure:
    """Create correlation timeline with regime annotations."""
    
    fig = make_subplots(
        rows=2 if show_tail_dep else 1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4] if show_tail_dep else [1.0],
        subplot_titles=['Rolling Correlation (ρ)', 'Lower Tail Dependence (λ<sub>L</sub>)'] if show_tail_dep else ['Rolling Correlation (ρ)']
    )
    
    colors = ['#c9a962', '#4a9ebb', '#22c55e', '#ef4444', '#a855f7', '#f97316']
    
    for idx, pair in enumerate(pairs):
        color = colors[idx % len(colors)]
        
        # Correlation
        corr_series = compute_rolling_correlation(returns, pair, window_size)
        fig.add_trace(go.Scatter(
            x=returns.index, y=corr_series,
            mode='lines', name=f'{pair[0]}/{pair[1]}',
            line=dict(color=color, width=2),
            hovertemplate=f'{pair[0]}/{pair[1]}<br>ρ=%{{y:.3f}}<br>%{{x}}<extra></extra>'
        ), row=1, col=1)
        
        # Detect and annotate regimes
        regimes = detect_regime_changes(corr_series)
        for regime in regimes[:5]:  # Limit annotations
            fig.add_vrect(
                x0=regime['start'], x1=regime['end'],
                fillcolor='rgba(239, 68, 68, 0.1)' if regime['type'] == 'spike' else 'rgba(34, 197, 94, 0.1)',
                line_width=0, row=1, col=1
            )
        
        # Tail dependence
        if show_tail_dep:
            tail_series = compute_rolling_tail_dependence(returns, pair, window_size)
            fig.add_trace(go.Scatter(
                x=returns.index, y=tail_series,
                mode='lines', name=f'{pair[0]}/{pair[1]} λ',
                line=dict(color=color, width=2, dash='dot'),
                hovertemplate=f'{pair[0]}/{pair[1]}<br>λ<sub>L</sub>=%{{y:.3f}}<br>%{{x}}<extra></extra>',
                showlegend=False
            ), row=2, col=1)
    
    # Add crisis annotations if provided
    if crisis_dates:
        for name, (start, end) in crisis_dates.items():
            try:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                if start_dt >= returns.index.min() and start_dt <= returns.index.max():
                    fig.add_vrect(
                        x0=start_dt, x1=end_dt,
                        fillcolor='rgba(239, 68, 68, 0.15)',
                        line=dict(color='#ef4444', width=1, dash='dash'),
                        row='all', col=1
                    )
                    fig.add_annotation(
                        x=start_dt, y=1.05,
                        text=name.split()[0],  # Short name
                        showarrow=False,
                        font=dict(size=9, color='#ef4444'),
                        xref='x', yref='y domain',
                        row=1, col=1
                    )
            except:
                pass
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(239,68,68,0.3)", row=1, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="rgba(34,197,94,0.3)", row=1, col=1)
    
    if show_tail_dep:
        fig.add_hline(y=0.3, line_dash="dot", line_color="rgba(239,68,68,0.3)", row=2, col=1)
    
    fig.update_layout(
        paper_bgcolor='#0f1117', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                   font=dict(size=11)),
        margin=dict(l=60, r=20, t=60, b=40),
        height=350 if show_tail_dep else 250,
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.2)', zeroline=False)
    
    # Update subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.color = '#f0f2f6'
        annotation.font.size = 12
    
    return fig


def create_conditional_distribution_plot(
    returns: pd.DataFrame,
    pair: tuple,
    window_size: int,
    condition_value: float
) -> go.Figure:
    """Create conditional distribution visualization."""
    
    window_data = returns.iloc[-window_size:]
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    
    # Get conditional distribution
    response_data, weights = compute_conditional_distribution(
        returns, pair, pair[0], condition_value, window_size, tolerance=0.5
    )
    
    # Create weighted histogram
    fig = go.Figure()
    
    # Unconditional distribution (background)
    fig.add_trace(go.Histogram(
        x=y,
        nbinsx=30,
        name='Unconditional',
        marker_color='rgba(75, 85, 99, 0.4)',
        opacity=0.5,
        histnorm='probability density'
    ))
    
    # Conditional distribution (highlighted)
    # Use weighted samples
    weighted_samples = np.repeat(response_data, (weights * 1000).astype(int))
    if len(weighted_samples) > 10:
        fig.add_trace(go.Histogram(
            x=weighted_samples,
            nbinsx=30,
            name=f'If {pair[0]}={condition_value:.1f}%',
            marker_color='#c9a962',
            opacity=0.8,
            histnorm='probability density'
        ))
    
    # Add vertical line for mean
    cond_mean = np.average(response_data, weights=weights)
    uncond_mean = np.mean(y)
    
    fig.add_vline(x=uncond_mean, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text=f"μ={uncond_mean:.2f}", annotation_font_color="#9ca3af")
    fig.add_vline(x=cond_mean, line_dash="solid", line_color="#c9a962",
                  annotation_text=f"μ|X={cond_mean:.2f}", annotation_font_color="#c9a962")
    
    fig.update_layout(
        title=dict(
            text=f'<b>Conditional Distribution</b>: {pair[1]} | {pair[0]} = {condition_value:.1f}%',
            font=dict(size=14, color='#f0f2f6')
        ),
        xaxis_title=f'{pair[1]} Returns (%)',
        yaxis_title='Density',
        paper_bgcolor='#0f1117',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=60, r=20, t=80, b=60),
        height=300,
        bargap=0.1,
        barmode='overlay'
    )
    
    fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.2)', zeroline=False)
    
    return fig




def generate_plotly_animation(
    returns: pd.DataFrame,
    pair: tuple,
    window_size: int,
    step_size: int
) -> go.Figure:
    """Generate Plotly animated 3D surface."""
    
    dates = returns.index
    frame_indices = list(range(window_size, len(returns), step_size))
    
    # Limit frames for performance
    if len(frame_indices) > 100:
        step_factor = len(frame_indices) // 100
        frame_indices = frame_indices[::step_factor]
    
    # Spectacular colorscale
    colorscale = [
        [0.0, 'rgba(15, 17, 23, 0.0)'],
        [0.05, 'rgba(20, 30, 50, 0.3)'],
        [0.15, 'rgba(30, 58, 95, 0.6)'],
        [0.3, 'rgba(45, 97, 135, 0.8)'],
        [0.45, 'rgba(74, 158, 187, 0.9)'],
        [0.6, 'rgba(126, 204, 232, 0.95)'],
        [0.75, 'rgba(184, 224, 240, 0.98)'],
        [0.88, 'rgba(232, 213, 163, 1.0)'],
        [1.0, 'rgba(201, 169, 98, 1.0)'],
    ]
    
    frames = []
    sliders_steps = []
    
    # Get global axis ranges
    all_x = returns[pair[0]].values * 100
    all_y = returns[pair[1]].values * 100
    xmin_global, xmax_global = np.percentile(all_x, [1, 99])
    ymin_global, ymax_global = np.percentile(all_y, [1, 99])
    xmin_global -= (xmax_global - xmin_global) * 0.3
    xmax_global += (xmax_global - xmin_global) * 0.3
    ymin_global -= (ymax_global - ymin_global) * 0.3
    ymax_global += (ymax_global - ymin_global) * 0.3
    
    grid_size = 60
    xx, yy = np.mgrid[xmin_global:xmax_global:complex(grid_size), 
                      ymin_global:ymax_global:complex(grid_size)]
    
    progress_bar = st.progress(0, text="Building animation frames...")
    
    for i, window_end in enumerate(frame_indices):
        window_start = max(0, window_end - window_size)
        window_data = returns.iloc[window_start:window_end]
        
        x = window_data[pair[0]].values * 100
        y = window_data[pair[1]].values * 100
        corr = np.corrcoef(x, y)[0, 1]
        tail_dep = compute_tail_dependence(x / 100, y / 100, u=0.05)
        date_str = dates[window_end].strftime('%Y-%m-%d')
        
        try:
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            density = gaussian_filter(density, sigma=1.0)
            density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)
        except:
            density_norm = np.zeros_like(xx)
        
        tail_str = f'λ={tail_dep:.2f}' if not np.isnan(tail_dep) else ''
        
        frame = go.Frame(
            data=[go.Surface(
                x=xx, y=yy, z=density_norm,
                colorscale=colorscale,
                showscale=False,
                opacity=0.95,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4),
                contours=dict(z=dict(show=True, usecolormap=True, project_z=False))
            )],
            name=str(i),
            layout=go.Layout(
                title=dict(
                    text=f'<b>{pair[0]} vs {pair[1]}</b>  ·  {date_str}  ·  ρ={corr:.2f}  {tail_str}',
                    font=dict(size=14, color='#f0f2f6')
                )
            )
        )
        frames.append(frame)
        
        sliders_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=100, redraw=True), mode='immediate')],
            label=date_str,
            method='animate'
        ))
        
        progress_bar.progress(min((i + 1) / len(frame_indices), 1.0), 
                             text=f"Building frame {i+1}/{len(frame_indices)}")
    
    progress_bar.empty()
    
    # Create initial figure with first frame data
    window_data = returns.iloc[:window_size]
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    
    try:
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        density = np.reshape(kernel(positions).T, xx.shape)
        density = gaussian_filter(density, sigma=1.0)
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)
    except:
        density_norm = np.zeros_like(xx)
    
    fig = go.Figure(
        data=[go.Surface(
            x=xx, y=yy, z=density_norm,
            colorscale=colorscale,
            showscale=False,
            opacity=0.95,
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False))
        )],
        frames=frames
    )
    
    # Add play/pause buttons and slider
    fig.update_layout(
        title=dict(
            text=f'<b>{pair[0]} vs {pair[1]}</b>  ·  Animated Evolution',
            x=0.5,
            font=dict(size=16, color='#f0f2f6')
        ),
        scene=dict(
            xaxis=dict(title=f'{pair[0]} (%)', title_font=dict(color='#9ca3af', size=11),
                      tickfont=dict(color='#6b7280', size=9), gridcolor='rgba(75, 85, 99, 0.15)',
                      showbackground=False, range=[xmin_global, xmax_global]),
            yaxis=dict(title=f'{pair[1]} (%)', title_font=dict(color='#9ca3af', size=11),
                      tickfont=dict(color='#6b7280', size=9), gridcolor='rgba(75, 85, 99, 0.15)',
                      showbackground=False, range=[ymin_global, ymax_global]),
            zaxis=dict(title='', showticklabels=False, showgrid=False, showbackground=False,
                      range=[0, 1]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            bgcolor='rgba(0,0,0,0)'
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=0))]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate',
                                           transition=dict(duration=0))])
                ],
                font=dict(color='#0f1117'),
                bgcolor='#c9a962'
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=12, color='#9ca3af'),
                prefix='Date: ',
                visible=True,
                xanchor='center'
            ),
            transition=dict(duration=100),
            pad=dict(b=10, t=50),
            len=0.8,
            x=0.1,
            y=0,
            steps=sliders_steps,
            font=dict(color='#6b7280'),
            tickcolor='#6b7280',
            bgcolor='#1a1d26',
            activebgcolor='#c9a962',
            bordercolor='#2a2f3a'
        )],
        paper_bgcolor='#0f1117',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=60, b=100),
        height=550
    )
    
    return fig


# ============================================================================
# DATA FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def download_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True,
                          progress=False, multi_level_index=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns."""
    return np.log(prices / prices.shift(1)).dropna()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">Preset Pairs</div>', unsafe_allow_html=True)
    
    preset = st.selectbox("Quick select", list(PRESET_PAIRS.keys()), index=0)
    
    st.markdown('<div class="sidebar-header">Asset Selection</div>', unsafe_allow_html=True)
    
    if preset != "Custom" and PRESET_PAIRS[preset]:
        preset_tickers = PRESET_PAIRS[preset]
        ticker1 = st.text_input("Ticker 1", value=preset_tickers[0], max_chars=10).upper().strip()
        ticker2 = st.text_input("Ticker 2", value=preset_tickers[1] if len(preset_tickers) > 1 else "", max_chars=10).upper().strip()
        ticker3 = st.text_input("Ticker 3 (optional)", value=preset_tickers[2] if len(preset_tickers) > 2 else "", max_chars=10).upper().strip()
        ticker4 = st.text_input("Ticker 4 (optional)", value="", max_chars=10).upper().strip()
    else:
        col1, col2 = st.columns(2)
        with col1:
            ticker1 = st.text_input("Ticker 1", value="SPY", max_chars=10).upper().strip()
            ticker3 = st.text_input("Ticker 3 (optional)", value="", max_chars=10).upper().strip()
        with col2:
            ticker2 = st.text_input("Ticker 2", value="TLT", max_chars=10).upper().strip()
            ticker4 = st.text_input("Ticker 4 (optional)", value="", max_chars=10).upper().strip()
    
    tickers = [t for t in [ticker1, ticker2, ticker3, ticker4] if t]
    tickers = list(dict.fromkeys(tickers))
    
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers")
        st.stop()
    
    st.markdown('<div class="sidebar-header">Date Range</div>', unsafe_allow_html=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    date_range = st.date_input("Select range", value=(start_date, end_date),
                               min_value=datetime(2000, 1, 1), max_value=end_date)
    
    if len(date_range) != 2:
        st.warning("Please select start and end dates")
        st.stop()
    
    st.markdown('<div class="sidebar-header">Rolling Window</div>', unsafe_allow_html=True)
    
    window_size = st.select_slider("Window size (trading days)",
                                   options=[30, 45, 60, 90, 120, 180, 252], value=60)
    
    st.markdown('<div class="sidebar-header">Analysis Options</div>', unsafe_allow_html=True)
    
    show_gaussian = st.checkbox("Show Gaussian overlay", value=False,
                                help="Compare empirical distribution to theoretical bivariate normal")
    
    show_tail_dep = st.checkbox("Show tail dependence", value=True,
                                help="Display λ_L alongside correlation")
    
    show_crisis = st.checkbox("Highlight crisis periods", value=False,
                              help="Mark major market crises on timeline")
    
    st.markdown('<div class="sidebar-header">Animation Settings</div>', unsafe_allow_html=True)
    
    step_size = st.slider("Step size (days per frame)", min_value=1, max_value=20, value=5,
                         help="Higher = faster animation, fewer frames")
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>Understanding the metrics:</strong><br><br>
        <b>ρ (rho)</b>: Linear correlation — average co-movement<br><br>
        <b>λ<sub>L</sub></b>: Lower tail dependence — P(Y crashes | X crashes)<br><br>
        ⚠️ High λ<sub>L</sub> with low ρ = "false diversifier"
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Download data
with st.spinner("Downloading market data..."):
    prices = download_data(tickers, date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d'))

if prices.empty:
    st.error("Could not download data. Please check ticker symbols.")
    st.stop()

returns = compute_returns(prices)
pairs = list(combinations(tickers, 2))
current_date = returns.index[-1].strftime('%Y-%m-%d')

# Compute current stats
window_data = returns.iloc[-window_size:]
stats_data = []
for pair in pairs:
    x = window_data[pair[0]].values
    y = window_data[pair[1]].values
    corr = np.corrcoef(x, y)[0, 1]
    tail_dep = compute_tail_dependence(x, y, u=0.05)
    stats_data.append({'pair': f"{pair[0]}/{pair[1]}", 'corr': corr, 'tail_dep': tail_dep})

# Display metrics
st.markdown(f"""
<div class="metric-card">
    <div style="display: flex; justify-content: space-around; text-align: center;">
        <div>
            <div class="metric-label">Assets</div>
            <div class="metric-value">{', '.join(tickers)}</div>
        </div>
        <div>
            <div class="metric-label">Pairs</div>
            <div class="metric-value">{len(pairs)}</div>
        </div>
        <div>
            <div class="metric-label">Data Points</div>
            <div class="metric-value">{len(returns):,}</div>
        </div>
        <div>
            <div class="metric-label">Window</div>
            <div class="metric-value">{window_size}d</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tail dependence warnings
for stat in stats_data:
    if not np.isnan(stat['tail_dep']) and stat['tail_dep'] > 0.3 and abs(stat['corr']) < 0.4:
        st.markdown(f"""
        <div class="tail-warning">
            ⚠️ <b>{stat['pair']}</b>: Low correlation (ρ={stat['corr']:.2f}) but HIGH tail dependence (λ<sub>L</sub>={stat['tail_dep']:.2f}) — potential false diversifier!
        </div>
        """, unsafe_allow_html=True)

# 3D Visualization
st.markdown("### Current Joint Distributions")
st.markdown("<p style='color: #6b7280; font-size: 0.85rem; margin-top: -10px;'>Drag to rotate · Scroll to zoom</p>", unsafe_allow_html=True)

if len(pairs) == 1:
    fig_plotly = create_plotly_3d_single(returns, pairs[0], window_size, current_date, show_gaussian)
else:
    fig_plotly = create_plotly_3d_grid(returns, pairs, window_size, current_date, show_gaussian)

st.plotly_chart(fig_plotly, use_container_width=True, config={'displayModeBar': False})

# Correlation Timeline
st.markdown("### Correlation & Tail Dependence Timeline")

crisis_to_show = CRISIS_DATES if show_crisis else None
fig_timeline = create_correlation_timeline(returns, pairs, window_size, crisis_to_show, show_tail_dep)
st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

# Conditional Distribution (What-If Slicer)
st.markdown("### What-If Analysis")
st.markdown("<p style='color: #6b7280; font-size: 0.85rem; margin-top: -10px;'>Explore conditional distributions: \"If Asset X moves by Y%, what happens to Asset Z?\"</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    if len(pairs) > 0:
        selected_pair_idx = st.selectbox("Select pair", range(len(pairs)), 
                                         format_func=lambda i: f"{pairs[i][0]} → {pairs[i][1]}")
        selected_pair = pairs[selected_pair_idx]
        
        x_data = returns.iloc[-window_size:][selected_pair[0]].values * 100
        x_min, x_max = float(np.percentile(x_data, 5)), float(np.percentile(x_data, 95))
        
        condition_value = st.slider(
            f"If {selected_pair[0]} returns:",
            min_value=x_min,
            max_value=x_max,
            value=float(np.percentile(x_data, 10)),  # Default to left tail
            step=0.1,
            format="%.1f%%"
        )

with col2:
    if len(pairs) > 0:
        fig_cond = create_conditional_distribution_plot(returns, selected_pair, window_size, condition_value)
        st.plotly_chart(fig_cond, use_container_width=True, config={'displayModeBar': False})

# Animation section
st.markdown("---")
st.markdown("### Animated Evolution")
st.markdown("<p style='color: #6b7280; font-size: 0.85rem; margin-top: -10px;'>Watch how the joint distribution evolves over time</p>", unsafe_allow_html=True)

if len(pairs) == 1:
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button("🎬 Generate Animation", use_container_width=True)
        st.markdown("""
        <div class="info-box" style="margin-top: 1rem;">
            <b>Tip:</b> Click Play to watch the distribution morph through time. Use the slider to jump to specific dates.
        </div>
        """, unsafe_allow_html=True)
    
    if 'animation_fig' not in st.session_state:
        st.session_state.animation_fig = None
    
    if generate_btn:
        st.session_state.animation_fig = generate_plotly_animation(
            returns, pairs[0], window_size, step_size
        )
    
    if st.session_state.animation_fig is not None:
        with col2:
            st.plotly_chart(st.session_state.animation_fig, use_container_width=True, 
                           config={'displayModeBar': False})
else:
    st.info("Animation is available for single pair analysis. Select 2 assets to enable animated view.")

# Footer
st.markdown("---")
st.markdown("""
<footer>
    <p>Built by <a href="https://twitter.com/Gsnchez" class="brand-link" target="_blank">@Gsnchez</a> · 
    <a href="https://bquantfinance.com" class="brand-link" target="_blank">bquantfinance.com</a></p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem; color: #6b7280;">
        Visualizing how asset relationships evolve through market regimes
    </p>
</footer>
""", unsafe_allow_html=True)
