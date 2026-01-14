"""
Joint Distribution Evolution Visualizer
@Gsnchez | bquantfinance.com

Visualize how the joint distribution of asset returns evolves over time.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.ndimage import gaussian_filter
from itertools import combinations
import io
import imageio
from datetime import datetime, timedelta
import plotly.graph_objects as go
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

def create_plotly_3d_single(
    returns: pd.DataFrame,
    pair: tuple,
    window_size: int,
    date_str: str
) -> go.Figure:
    """Create spectacular interactive 3D Plotly surface for a single pair."""
    
    window_data = returns.iloc[-window_size:]
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    corr = np.corrcoef(x, y)[0, 1]
    
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
        [0.0, '#0f1117'],
        [0.15, '#1a1f2e'],
        [0.3, '#1e3a5f'],
        [0.45, '#2d6187'],
        [0.6, '#4a9ebb'],
        [0.75, '#7ecce8'],
        [0.85, '#b8e0f0'],
        [0.95, '#e8d5a3'],
        [1.0, '#c9a962'],
    ]
    
    # Create 3D surface
    fig = go.Figure()
    
    # Main surface
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=density_norm,
        colorscale=colorscale,
        showscale=False,
        opacity=0.95,
        lighting=dict(
            ambient=0.4,
            diffuse=0.6,
            specular=0.3,
            roughness=0.5,
            fresnel=0.2
        ),
        lightposition=dict(x=100, y=100, z=100),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="#c9a962",
                project_z=True
            )
        ),
        hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<br>Density: %{{z:.3f}}<extra></extra>'
    ))
    
    # Scatter points on the bottom
    z_offset = -0.1
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=np.full_like(x, z_offset),
        mode='markers',
        marker=dict(
            size=3,
            color='#c9a962',
            opacity=0.7,
            symbol='circle'
        ),
        hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<extra></extra>',
        name='Returns'
    ))
    
    # Correlation color
    corr_color = '#ef4444' if corr > 0.5 else '#22c55e' if corr < -0.2 else '#c9a962'
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>{pair[0]} vs {pair[1]}</b>  ·  {date_str}  ·  <span style="color:{corr_color}">ρ = {corr:.2f}</span>',
            x=0.5,
            font=dict(size=16, color='#f0f2f6')
        ),
        scene=dict(
            xaxis=dict(
                title=f'{pair[0]} (%)',
                titlefont=dict(color='#9ca3af', size=12),
                tickfont=dict(color='#6b7280', size=10),
                gridcolor='rgba(75, 85, 99, 0.3)',
                backgroundcolor='#0f1117',
                showbackground=True,
                zerolinecolor='#2a2f3a'
            ),
            yaxis=dict(
                title=f'{pair[1]} (%)',
                titlefont=dict(color='#9ca3af', size=12),
                tickfont=dict(color='#6b7280', size=10),
                gridcolor='rgba(75, 85, 99, 0.3)',
                backgroundcolor='#0f1117',
                showbackground=True,
                zerolinecolor='#2a2f3a'
            ),
            zaxis=dict(
                title='Density',
                titlefont=dict(color='#9ca3af', size=12),
                tickfont=dict(color='#6b7280', size=10),
                gridcolor='rgba(75, 85, 99, 0.3)',
                backgroundcolor='#0f1117',
                showbackground=True,
                showticklabels=False
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6)
        ),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        height=550
    )
    
    return fig


def create_plotly_3d_grid(
    returns: pd.DataFrame,
    pairs: list,
    window_size: int,
    date_str: str
) -> go.Figure:
    """Create spectacular interactive 3D Plotly surfaces for multiple pairs."""
    from plotly.subplots import make_subplots
    
    n_pairs = len(pairs)
    
    if n_pairs <= 3:
        rows, cols = 1, n_pairs
    else:
        rows, cols = 2, 3
    
    # Create subplot specs for 3D
    specs = [[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=[f'{p[0]} vs {p[1]}' for p in pairs],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    # Spectacular colorscale
    colorscale = [
        [0.0, '#0f1117'],
        [0.15, '#1a1f2e'],
        [0.3, '#1e3a5f'],
        [0.45, '#2d6187'],
        [0.6, '#4a9ebb'],
        [0.75, '#7ecce8'],
        [0.85, '#b8e0f0'],
        [0.95, '#e8d5a3'],
        [1.0, '#c9a962'],
    ]
    
    window_data = returns.iloc[-window_size:]
    
    for idx, pair in enumerate(pairs):
        row = idx // cols + 1
        col = idx % cols + 1
        
        x = window_data[pair[0]].values * 100
        y = window_data[pair[1]].values * 100
        corr = np.corrcoef(x, y)[0, 1]
        
        # KDE
        try:
            xmin, xmax = np.percentile(x, [2, 98])
            ymin, ymax = np.percentile(y, [2, 98])
            x_range = xmax - xmin
            y_range = ymax - ymin
            xmin -= x_range * 0.3
            xmax += x_range * 0.3
            ymin -= y_range * 0.3
            ymax += y_range * 0.3
            
            grid_size = 70
            xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            density = gaussian_filter(density, sigma=1.0)
            density_norm = (density - density.min()) / (density.max() - density.min())
        except:
            xx, yy = np.mgrid[-3:3:50j, -3:3:50j]
            density_norm = np.zeros_like(xx)
        
        # Add surface
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=density_norm,
                colorscale=colorscale,
                showscale=False,
                opacity=0.95,
                lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3),
                hovertemplate=f'{pair[0]}: %{{x:.2f}}%<br>{pair[1]}: %{{y:.2f}}%<extra>ρ={corr:.2f}</extra>'
            ),
            row=row, col=col
        )
        
        # Add scatter points
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=np.full_like(x, -0.1),
                mode='markers',
                marker=dict(size=2, color='#c9a962', opacity=0.6),
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Update scene for this subplot
        scene_name = f'scene{idx + 1}' if idx > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(
                    title=f'{pair[0]} (%)',
                    titlefont=dict(color='#9ca3af', size=10),
                    tickfont=dict(color='#6b7280', size=8),
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    backgroundcolor='#0f1117',
                    showbackground=True
                ),
                yaxis=dict(
                    title=f'{pair[1]} (%)',
                    titlefont=dict(color='#9ca3af', size=10),
                    tickfont=dict(color='#6b7280', size=8),
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    backgroundcolor='#0f1117',
                    showbackground=True
                ),
                zaxis=dict(
                    title='',
                    showticklabels=False,
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    backgroundcolor='#0f1117',
                    showbackground=True
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            )
        })
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Joint Distributions</b>  ·  {date_str}',
            x=0.5,
            font=dict(size=18, color='#f0f2f6')
        ),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=False,
        height=500 if n_pairs <= 3 else 800
    )
    
    # Update subplot title colors
    for annotation in fig.layout.annotations:
        annotation.font.color = '#f0f2f6'
        annotation.font.size = 12
    
    return fig


# ============================================================================
# DATA FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def download_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices for selected tickers."""
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )['Close']
        
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def create_joint_density_frame(
    returns: pd.DataFrame,
    pair: tuple,
    window_end: int,
    window_size: int,
    date_str: str,
    figsize: tuple = (8, 7),
    azimuth: float = -60
) -> Figure:
    """Create a spectacular 3D surface plot of the joint density."""
    
    window_start = max(0, window_end - window_size)
    window_data = returns.iloc[window_start:window_end]
    
    x = window_data[pair[0]].values * 100
    y = window_data[pair[1]].values * 100
    
    # Compute correlation
    corr = np.corrcoef(x, y)[0, 1]
    
    # Create figure with dark background
    fig = plt.figure(figsize=figsize, facecolor='#0f1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0f1117')
    
    try:
        # KDE for smooth density
        xmin, xmax = np.percentile(x, [1, 99])
        ymin, ymax = np.percentile(y, [1, 99])
        
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= x_range * 0.3
        xmax += x_range * 0.3
        ymin -= y_range * 0.3
        ymax += y_range * 0.3
        
        # Higher resolution grid for smooth surface
        xx, yy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        density = np.reshape(kernel(positions).T, xx.shape)
        
        # Normalize density for better visualization
        density_norm = (density - density.min()) / (density.max() - density.min())
        
        # Custom colormap - dark blue to cyan to gold
        from matplotlib.colors import LinearSegmentedColormap
        colors_3d = [
            '#0f1117',  # Dark base
            '#1a1f2e',  # Dark blue
            '#1e3a5f',  # Navy
            '#2d6187',  # Steel blue
            '#4a9ebb',  # Cyan
            '#7ecce8',  # Light cyan
            '#b8e0f0',  # Pale cyan
            '#e8d5a3',  # Warm highlight
            '#c9a962',  # Gold peak
        ]
        cmap_3d = LinearSegmentedColormap.from_list('spectacular', colors_3d)
        
        # Plot 3D surface
        surf = ax.plot_surface(
            xx, yy, density_norm,
            cmap=cmap_3d,
            edgecolor='none',
            alpha=0.95,
            antialiased=True,
            rstride=1,
            cstride=1,
            shade=True
        )
        
        # Add contour projection on the bottom
        offset = -0.15
        ax.contourf(
            xx, yy, density_norm,
            zdir='z',
            offset=offset,
            cmap=cmap_3d,
            alpha=0.4,
            levels=15
        )
        
        # Add subtle contour lines on surface
        ax.contour(
            xx, yy, density_norm,
            zdir='z',
            offset=offset,
            colors='#c9a962',
            alpha=0.3,
            linewidths=0.5,
            levels=8
        )
        
        # Scatter points projection on bottom
        z_scatter = np.full_like(x, offset + 0.01)
        ax.scatter(x, y, z_scatter, c='#c9a962', s=8, alpha=0.6, edgecolors='none')
        
    except Exception as e:
        # Fallback
        ax.scatter(x, y, np.zeros_like(x), c='#c9a962', s=15, alpha=0.7)
    
    # Styling
    ax.set_xlabel(f'{pair[0]} (%)', fontsize=10, color='#9ca3af', labelpad=10)
    ax.set_ylabel(f'{pair[1]} (%)', fontsize=10, color='#9ca3af', labelpad=10)
    ax.set_zlabel('Density', fontsize=10, color='#9ca3af', labelpad=10)
    
    # Set view angle - use passed azimuth for rotation effect
    ax.view_init(elev=25, azim=azimuth)
    
    # Style the panes
    ax.xaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
    ax.yaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
    ax.zaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
    
    # Style the grid
    ax.xaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.3)
    ax.yaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.3)
    ax.zaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.3)
    
    # Tick colors
    ax.tick_params(axis='x', colors='#6b7280', labelsize=8)
    ax.tick_params(axis='y', colors='#6b7280', labelsize=8)
    ax.tick_params(axis='z', colors='#6b7280', labelsize=8)
    
    # Title
    corr_color = '#ef4444' if corr > 0.5 else '#22c55e' if corr < -0.2 else '#c9a962'
    fig.suptitle(
        f'{pair[0]} vs {pair[1]}  ·  {date_str}  ·  ρ = {corr:.2f}',
        fontsize=13,
        color='#f0f2f6',
        fontweight='600',
        y=0.95
    )
    
    # Remove z-axis ticks for cleaner look
    ax.set_zticks([])
    
    plt.tight_layout()
    return fig


def create_grid_frame(
    returns: pd.DataFrame,
    pairs: list,
    window_end: int,
    window_size: int,
    date_str: str,
    azimuth: float = -60
) -> Figure:
    """Create a grid of spectacular 3D joint density plots for multiple pairs."""
    
    n_pairs = len(pairs)
    
    if n_pairs == 1:
        ncols, nrows = 1, 1
        figsize = (9, 8)
    elif n_pairs <= 3:
        ncols, nrows = n_pairs, 1
        figsize = (6 * n_pairs, 6)
    else:
        ncols, nrows = 3, 2
        figsize = (16, 11)
    
    fig = plt.figure(figsize=figsize, facecolor='#0f1117')
    
    window_start = max(0, window_end - window_size)
    window_data = returns.iloc[window_start:window_end]
    
    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_3d = [
        '#0f1117',
        '#1a1f2e',
        '#1e3a5f',
        '#2d6187',
        '#4a9ebb',
        '#7ecce8',
        '#b8e0f0',
        '#e8d5a3',
        '#c9a962',
    ]
    cmap_3d = LinearSegmentedColormap.from_list('spectacular', colors_3d)
    
    for idx, pair in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d', facecolor='#0f1117')
        
        x = window_data[pair[0]].values * 100
        y = window_data[pair[1]].values * 100
        
        corr = np.corrcoef(x, y)[0, 1]
        
        try:
            xmin, xmax = np.percentile(x, [2, 98])
            ymin, ymax = np.percentile(y, [2, 98])
            
            x_range = xmax - xmin
            y_range = ymax - ymin
            xmin -= x_range * 0.3
            xmax += x_range * 0.3
            ymin -= y_range * 0.3
            ymax += y_range * 0.3
            
            # Grid for surface
            xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            
            # Normalize
            density_norm = (density - density.min()) / (density.max() - density.min())
            
            # 3D surface
            ax.plot_surface(
                xx, yy, density_norm,
                cmap=cmap_3d,
                edgecolor='none',
                alpha=0.95,
                antialiased=True,
                rstride=2,
                cstride=2,
                shade=True
            )
            
            # Contour projection on bottom
            offset = -0.15
            ax.contourf(
                xx, yy, density_norm,
                zdir='z',
                offset=offset,
                cmap=cmap_3d,
                alpha=0.35,
                levels=12
            )
            
            # Scatter points
            z_scatter = np.full_like(x, offset + 0.01)
            ax.scatter(x, y, z_scatter, c='#c9a962', s=6, alpha=0.5, edgecolors='none')
            
        except Exception:
            ax.scatter(x, y, np.zeros_like(x), c='#c9a962', s=10, alpha=0.6)
        
        # Styling
        ax.set_xlabel(f'{pair[0]} (%)', fontsize=9, color='#9ca3af', labelpad=8)
        ax.set_ylabel(f'{pair[1]} (%)', fontsize=9, color='#9ca3af', labelpad=8)
        
        ax.view_init(elev=25, azim=azimuth)
        
        # Pane colors
        ax.xaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
        ax.yaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
        ax.zaxis.set_pane_color((0.06, 0.07, 0.09, 1.0))
        
        # Grid colors
        ax.xaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.25)
        ax.yaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.25)
        ax.zaxis._axinfo['grid']['color'] = (0.3, 0.35, 0.4, 0.25)
        
        ax.tick_params(axis='x', colors='#6b7280', labelsize=7)
        ax.tick_params(axis='y', colors='#6b7280', labelsize=7)
        ax.set_zticks([])
        
        # Title per subplot
        corr_color = '#ef4444' if corr > 0.5 else '#22c55e' if corr < -0.2 else '#c9a962'
        ax.set_title(
            f'{pair[0]} vs {pair[1]}  ·  ρ = {corr:.2f}',
            fontsize=11,
            color='#f0f2f6',
            fontweight='500',
            pad=5,
            y=1.0
        )
    
    # Main title
    fig.suptitle(
        f'{date_str}',
        fontsize=15,
        color='#f0f2f6',
        fontweight='600',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


def generate_gif(
    returns: pd.DataFrame,
    pairs: list,
    window_size: int,
    step_size: int,
    fps: int = 5,
    rotate: bool = False
) -> bytes:
    """Generate animated GIF of evolving joint distributions."""
    
    frames = []
    dates = returns.index
    
    # Calculate actual frame indices
    frame_indices = list(range(window_size, len(returns), step_size))
    total_frames = len(frame_indices)
    
    progress_bar = st.progress(0, text="Generating animation...")
    
    for i, window_end in enumerate(frame_indices):
        date_str = dates[window_end].strftime('%Y-%m-%d')
        
        # Calculate azimuth for rotation effect (subtle 45-degree sweep)
        if rotate:
            azimuth = -60 + (i / total_frames) * 45
        else:
            azimuth = -60
        
        if len(pairs) == 1:
            fig = create_joint_density_frame(
                returns, pairs[0], window_end, window_size, date_str, azimuth=azimuth
            )
        else:
            fig = create_grid_frame(
                returns, pairs, window_end, window_size, date_str, azimuth=azimuth
            )
        
        # Convert figure to image - higher DPI for spectacular quality
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor='#0f1117', 
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
        
        progress_bar.progress(min((i + 1) / total_frames, 1.0), text=f"Generating frame {i+1}/{total_frames}")
    
    progress_bar.empty()
    
    # Create GIF
    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, frames, format='GIF', fps=fps, loop=0)
    gif_buffer.seek(0)
    
    return gif_buffer.getvalue()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">Asset Selection</div>', unsafe_allow_html=True)
    
    # Default tickers
    default_tickers = ["SPY", "TLT", "GLD", "QQQ"]
    
    # Ticker inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input("Ticker 1", value="SPY", max_chars=10).upper().strip()
        ticker3 = st.text_input("Ticker 3 (optional)", value="GLD", max_chars=10).upper().strip()
    with col2:
        ticker2 = st.text_input("Ticker 2", value="TLT", max_chars=10).upper().strip()
        ticker4 = st.text_input("Ticker 4 (optional)", value="", max_chars=10).upper().strip()
    
    # Build ticker list
    tickers = [t for t in [ticker1, ticker2, ticker3, ticker4] if t]
    tickers = list(dict.fromkeys(tickers))  # Remove duplicates, preserve order
    
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers")
        st.stop()
    
    st.markdown('<div class="sidebar-header">Date Range</div>', unsafe_allow_html=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    date_range = st.date_input(
        "Select range",
        value=(start_date, end_date),
        min_value=datetime(2000, 1, 1),
        max_value=end_date
    )
    
    if len(date_range) != 2:
        st.warning("Please select start and end dates")
        st.stop()
    
    st.markdown('<div class="sidebar-header">Rolling Window</div>', unsafe_allow_html=True)
    
    window_size = st.select_slider(
        "Window size (trading days)",
        options=[30, 45, 60, 90, 120, 180, 252],
        value=60
    )
    
    st.markdown('<div class="sidebar-header">Animation Settings</div>', unsafe_allow_html=True)
    
    step_size = st.slider(
        "Step size (days per frame)",
        min_value=1,
        max_value=20,
        value=5,
        help="Higher = faster animation, fewer frames"
    )
    
    fps = st.slider(
        "Frames per second",
        min_value=2,
        max_value=15,
        value=5
    )
    
    rotate_camera = st.checkbox(
        "Rotate camera",
        value=False,
        help="Slowly rotate the 3D view during animation"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong><br>
        1. Select 2-4 assets<br>
        2. Choose date range<br>
        3. Click "Generate Animation"<br>
        4. Export GIF to share
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Download data
with st.spinner("Downloading market data..."):
    prices = download_data(
        tickers,
        date_range[0].strftime('%Y-%m-%d'),
        date_range[1].strftime('%Y-%m-%d')
    )

if prices.empty:
    st.error("Could not download data. Please check ticker symbols.")
    st.stop()

# Compute returns
returns = compute_returns(prices)

# Generate pairs
pairs = list(combinations(tickers, 2))

# Display info
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

# Current snapshot
st.markdown("### Current Joint Distributions")
st.markdown("<p style='color: #6b7280; font-size: 0.85rem; margin-top: -10px;'>Drag to rotate · Scroll to zoom</p>", unsafe_allow_html=True)

# Show current state with interactive Plotly
current_date = returns.index[-1].strftime('%Y-%m-%d')

if len(pairs) == 1:
    fig_plotly = create_plotly_3d_single(returns, pairs[0], window_size, current_date)
    st.plotly_chart(fig_plotly, use_container_width=True, config={'displayModeBar': False})
else:
    fig_plotly = create_plotly_3d_grid(returns, pairs, window_size, current_date)
    st.plotly_chart(fig_plotly, use_container_width=True, config={'displayModeBar': False})

# Animation section
st.markdown("---")
st.markdown("### Generate Animation")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    generate_btn = st.button("🎬 Generate Animation", use_container_width=True)

# Store GIF in session state
if 'gif_data' not in st.session_state:
    st.session_state.gif_data = None

if generate_btn:
    with st.spinner("Creating animation... This may take a minute."):
        st.session_state.gif_data = generate_gif(
            returns, pairs, window_size, step_size, fps, rotate=rotate_camera
        )
    st.success("Animation ready!")

# Display and download
if st.session_state.gif_data is not None:
    st.markdown("### Preview")
    st.image(st.session_state.gif_data, use_container_width=True)
    
    # Download button
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"joint_dist_{'_'.join(tickers)}_{timestamp}.gif"
    
    st.download_button(
        label="📥 Download GIF",
        data=st.session_state.gif_data,
        file_name=filename,
        mime="image/gif",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================
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
