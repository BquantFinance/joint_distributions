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
from scipy import stats
from itertools import combinations
import io
import imageio
from datetime import datetime, timedelta
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

# Custom CSS for dark aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0a0a0a;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid #1a1a2e;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    .brand-link {
        color: #4a9eff;
        text-decoration: none;
        font-weight: 500;
    }
    
    .brand-link:hover {
        color: #7ab8ff;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #2a2a4a;
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #8892a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .info-box {
        background: #1a1a2e;
        border-left: 3px solid #4a9eff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #b0b0b0;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1a2e;
        border-color: #2a2a4a;
    }
    
    .stSlider > div > div {
        background-color: #4a9eff;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0d0d14;
        border-right: 1px solid #1a1a2e;
    }
    
    .sidebar-header {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    footer {
        text-align: center;
        padding: 2rem 0;
        color: #4a4a6a;
        font-size: 0.85rem;
        border-top: 1px solid #1a1a2e;
        margin-top: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff 0%, #2d7dd2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5aa8ff 0%, #3d8de2 100%);
        box-shadow: 0 4px 15px rgba(74, 158, 255, 0.3);
    }
    
    .download-btn {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
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
        <a href="https://twitter.com/Gsnchez" class="brand-link" target="_blank">@Gsnchez</a> | 
        <a href="https://bquantfinance.com" class="brand-link" target="_blank">bquantfinance.com</a>
    </div>
</div>
""", unsafe_allow_html=True)

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
        
        # Ensure column names are simple strings
        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)
        
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
    figsize: tuple = (6, 5)
) -> Figure:
    """Create a single frame of the joint density plot."""
    
    window_start = max(0, window_end - window_size)
    window_data = returns.iloc[window_start:window_end]
    
    x = window_data[pair[0]].values * 100  # Convert to percentage
    y = window_data[pair[1]].values * 100
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Compute correlation
    corr = np.corrcoef(x, y)[0, 1]
    
    # Create 2D histogram / density
    try:
        # KDE for smooth density
        xmin, xmax = np.percentile(x, [1, 99])
        ymin, ymax = np.percentile(y, [1, 99])
        
        # Expand range slightly
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= x_range * 0.2
        xmax += x_range * 0.2
        ymin -= y_range * 0.2
        ymax += y_range * 0.2
        
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        density = np.reshape(kernel(positions).T, xx.shape)
        
        # Contour plot
        levels = np.linspace(density.min(), density.max(), 15)
        contour = ax.contourf(xx, yy, density, levels=levels, cmap='Blues', alpha=0.8)
        ax.contour(xx, yy, density, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)
        
    except Exception:
        # Fallback to scatter if KDE fails
        ax.scatter(x, y, alpha=0.5, s=10, c='#4a9eff', edgecolors='none')
    
    # Scatter points on top
    ax.scatter(x, y, alpha=0.4, s=8, c='white', edgecolors='none', zorder=5)
    
    # Style axes
    ax.set_xlabel(f'{pair[0]} Returns (%)', fontsize=10, color='#b0b0b0', fontweight='500')
    ax.set_ylabel(f'{pair[1]} Returns (%)', fontsize=10, color='#b0b0b0', fontweight='500')
    
    ax.tick_params(colors='#6c757d', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2a4a')
        spine.set_linewidth(0.5)
    
    ax.axhline(0, color='#3a3a5a', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='#3a3a5a', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # Title with date and correlation
    corr_color = '#ff6b6b' if corr > 0.5 else '#4ecdc4' if corr < -0.2 else '#f9ca24'
    ax.set_title(
        f'{date_str}  |  ρ = {corr:.2f}',
        fontsize=11,
        color='white',
        fontweight='600',
        pad=10
    )
    
    # Add correlation color indicator
    ax.add_patch(plt.Rectangle(
        (0.02, 0.92), 0.04, 0.04,
        transform=ax.transAxes,
        facecolor=corr_color,
        edgecolor='none',
        zorder=10
    ))
    
    plt.tight_layout()
    return fig


def create_grid_frame(
    returns: pd.DataFrame,
    pairs: list,
    window_end: int,
    window_size: int,
    date_str: str
) -> Figure:
    """Create a grid of joint density plots for multiple pairs."""
    
    n_pairs = len(pairs)
    
    if n_pairs == 1:
        ncols, nrows = 1, 1
        figsize = (7, 6)
    elif n_pairs <= 3:
        ncols, nrows = n_pairs, 1
        figsize = (5 * n_pairs, 5)
    else:
        ncols, nrows = 3, 2
        figsize = (14, 10)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#0a0a0a')
    
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    window_start = max(0, window_end - window_size)
    window_data = returns.iloc[window_start:window_end]
    
    for idx, (pair, ax) in enumerate(zip(pairs, axes)):
        ax.set_facecolor('#0a0a0a')
        
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
            
            xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            
            levels = np.linspace(density.min(), density.max(), 12)
            ax.contourf(xx, yy, density, levels=levels, cmap='Blues', alpha=0.8)
            ax.contour(xx, yy, density, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)
            
        except Exception:
            ax.scatter(x, y, alpha=0.5, s=8, c='#4a9eff', edgecolors='none')
        
        ax.scatter(x, y, alpha=0.3, s=5, c='white', edgecolors='none', zorder=5)
        
        ax.set_xlabel(f'{pair[0]} (%)', fontsize=9, color='#b0b0b0')
        ax.set_ylabel(f'{pair[1]} (%)', fontsize=9, color='#b0b0b0')
        ax.tick_params(colors='#6c757d', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')
            spine.set_linewidth(0.5)
        
        ax.axhline(0, color='#3a3a5a', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='#3a3a5a', linewidth=0.5, linestyle='--', alpha=0.5)
        
        corr_color = '#ff6b6b' if corr > 0.5 else '#4ecdc4' if corr < -0.2 else '#f9ca24'
        ax.set_title(
            f'{pair[0]} vs {pair[1]}  |  ρ = {corr:.2f}',
            fontsize=10,
            color='white',
            fontweight='500',
            pad=8
        )
    
    # Hide unused axes
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    
    # Main title
    fig.suptitle(
        f'{date_str}',
        fontsize=14,
        color='white',
        fontweight='600',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def generate_gif(
    returns: pd.DataFrame,
    pairs: list,
    window_size: int,
    step_size: int,
    fps: int = 5
) -> bytes:
    """Generate animated GIF of evolving joint distributions."""
    
    frames = []
    dates = returns.index
    
    total_frames = (len(returns) - window_size) // step_size
    progress_bar = st.progress(0, text="Generating animation...")
    
    for i, window_end in enumerate(range(window_size, len(returns), step_size)):
        date_str = dates[window_end].strftime('%Y-%m-%d')
        
        if len(pairs) == 1:
            fig = create_joint_density_frame(
                returns, pairs[0], window_end, window_size, date_str
            )
        else:
            fig = create_grid_frame(
                returns, pairs, window_end, window_size, date_str
            )
        
        # Convert figure to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='#0a0a0a', 
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
        
        progress_bar.progress((i + 1) / total_frames, text=f"Generating frame {i+1}/{total_frames}")
    
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

# Show current state
current_date = returns.index[-1].strftime('%Y-%m-%d')

if len(pairs) == 1:
    fig = create_joint_density_frame(
        returns, pairs[0], len(returns), window_size, current_date, figsize=(8, 6)
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    fig = create_grid_frame(returns, pairs, len(returns), window_size, current_date)
    st.pyplot(fig)
    plt.close(fig)

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
            returns, pairs, window_size, step_size, fps
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
    <p>Built with 💙 by <a href="https://twitter.com/Gsnchez" class="brand-link" target="_blank">@Gsnchez</a> | 
    <a href="https://bquantfinance.com" class="brand-link" target="_blank">BQuant Finance</a></p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        Visualizing how asset relationships evolve through market regimes.
    </p>
</footer>
""", unsafe_allow_html=True)
