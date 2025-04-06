import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from streamlit_option_menu import option_menu
from PIL import Image

# ================================================
# SETUP & CONFIGURATION
# ================================================

# Page config
st.set_page_config(
    page_title="DataInsight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create style.css file if it doesn't exist
css_content = """
:root {
    --primary-color: #1E3A8A;
    --secondary-color: #3B82F6;
    --accent-color: #10B981;
    --background-color: #F9FAFB;
    --text-color: #111827;
    --card-bg: #FFFFFF;
}

[data-theme="dark"] {
    --primary-color: #3B82F6;
    --secondary-color: #1E40AF;
    --accent-color: #10B981;
    --background-color: #0F172A;
    --text-color: #E5E7EB;
    --card-bg: #1E293B;
}

.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

.stMetric {
    background-color: var(--card-bg) !important;
    border-radius: 10px;
    padding: 15px;
}

.custom-card {
    background-color: var(--card-bg);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
"""

with open("style.css", "w") as f:
    f.write(css_content)

load_css()

# ================================================
# UTILITY FUNCTIONS
# ================================================

@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel file into DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def clean_data(df):
    """Perform data cleaning operations"""
    # Convert columns to appropriate types
    for col in df.columns:
        # Try to convert to datetime
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except (ValueError, TypeError):
            pass
        
        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except ValueError:
            pass
    
    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def get_excel_download_link(df):
    """Generate Excel download link"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    processed_data = output.getvalue()
    return processed_data

# ================================================
# VISUALIZATION FUNCTIONS
# ================================================

def plot_advanced_distribution(df, column):
    """Advanced distribution plot with multiple views"""
    tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
    
    with tab1:
        fig = px.histogram(
            df, 
            x=column, 
            marginal="box",
            title=f"Distribution of {column}",
            color_discrete_sequence=[st.get_option("theme.primaryColor")]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.box(
            df, 
            y=column,
            title=f"Box Plot of {column}",
            color_discrete_sequence=[st.get_option("theme.secondaryColor")]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.violin(
            df, 
            y=column,
            title=f"Violin Plot of {column}",
            box=True,
            color_discrete_sequence=[st.get_option("theme.accentColor")]
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df):
    """Interactive correlation matrix with heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation plot")
        return
    
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series_analysis(df, date_col, value_col):
    """Advanced time series analysis"""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    tab1, tab2, tab3 = st.tabs(["Line Chart", "Seasonality", "Decomposition"])
    
    with tab1:
        fig = px.line(
            df, 
            x=date_col, 
            y=value_col,
            title=f"{value_col} over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        try:
            df['month'] = df[date_col].dt.month
            monthly_avg = df.groupby('month')[value_col].mean().reset_index()
            
            fig = px.bar(
                monthly_avg,
                x='month',
                y=value_col,
                title=f"Monthly Seasonality of {value_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not analyze seasonality: {e}")
    
    with tab3:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            df = df.set_index(date_col).sort_index()
            result = seasonal_decompose(df[value_col], model='additive', period=12)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
            result.observed.plot(ax=ax1, title='Observed')
            result.trend.plot(ax=ax2, title='Trend')
            result.seasonal.plot(ax=ax3, title='Seasonal')
            result.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not decompose time series: {e}")

def plot_multivariate_analysis(df, x_col, y_col, hue_col=None):
    """Multivariate analysis with different plot types"""
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Bubble Chart", "Parallel Coordinates"])
    
    with tab1:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            title=f"{y_col} vs {x_col}",
            trendline="lowess"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if hue_col:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=y_col,
                color=hue_col,
                hover_name=hue_col,
                size_max=30,
                title=f"Bubble Chart: {y_col} vs {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select a hue column for bubble chart")
    
    with tab3:
        if len(df.select_dtypes(include=[np.number]).columns) > 2:
            fig = px.parallel_coordinates(
                df,
                color=x_col if x_col in df.select_dtypes(include=[np.number]).columns else y_col,
                title="Parallel Coordinates Plot"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numeric columns for parallel coordinates")

# ================================================
# PAGE COMPONENTS
# ================================================

def home_page():
    """Home page with file upload and app information"""
    st.title("📊 DataInsight Pro")
    st.markdown("""
    <div class="custom-card">
        <h3>Advanced Data Analytics Platform</h3>
        <p>Upload your dataset to uncover powerful insights with our automated analysis tools.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Drag & drop your CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['raw_data'] = df
                st.session_state['clean_data'] = clean_data(df)
                st.success("Data loaded successfully!")
    
    with col2:
        st.subheader("Quick Start Guide")
        with st.expander("How to use this tool"):
            st.markdown("""
            1. **Upload** your data file (CSV or Excel)
            2. Navigate to **Summary** for data overview
            3. Explore **Visualizations** for insights
            4. Export results from any page
            """)
        
        st.subheader("Sample Datasets")
        if st.button("Load Sample Dataset (Iris)"):
            st.session_state['raw_data'] = sns.load_dataset('iris')
            st.session_state['clean_data'] = st.session_state['raw_data'].copy()
            st.success("Sample dataset loaded!")

def summary_page():
    """Data summary and statistics page"""
    if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
        st.warning("Please upload data from the Home page")
        return
    
    df = st.session_state['clean_data']
    
    st.title("📋 Data Summary")
    st.markdown("""
    <div class="custom-card">
        Comprehensive overview of your dataset's structure and statistics
    </div>
    """, unsafe_allow_html=True)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Unique Values': df.nunique(),
        'Missing Values': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include='all'), use_container_width=True)

def visuals_page():
    """Interactive visualizations page"""
    if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
        st.warning("Please upload data from the Home page")
        return
    
    df = st.session_state['clean_data']
    
    st.title("📈 Advanced Visualizations")
    st.markdown("""
    <div class="custom-card">
        Interactive visualizations to explore your data from multiple perspectives
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Analysis", "Correlation Analysis", 
         "Time Series Analysis", "Multivariate Analysis"]
    )
    
    if viz_type == "Distribution Analysis":
        st.subheader("Distribution Analysis")
        column = st.selectbox("Select column", df.columns)
        plot_advanced_distribution(df, column)
    
    elif viz_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        plot_correlation_matrix(df)
    
    elif viz_type == "Time Series Analysis":
        st.subheader("Time Series Analysis")
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0:
            date_col = st.selectbox("Select date column", date_cols)
            value_col = st.selectbox("Select value column", df.select_dtypes(include=[np.number]).columns)
            plot_time_series_analysis(df, date_col, value_col)
        else:
            st.warning("No datetime columns found for time series analysis")
    
    elif viz_type == "Multivariate Analysis":
        st.subheader("Multivariate Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis column", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            hue_col = st.selectbox("Color by (optional)", [None] + list(df.columns))
            plot_multivariate_analysis(df, x_col, y_col, hue_col)
        else:
            st.warning("Need at least 2 numeric columns for multivariate analysis")

def about_page():
    """About page with application information"""
    st.title("ℹ️ About DataInsight Pro")
    
    st.markdown("""
    <div class="custom-card">
        <h3>Advanced Data Analytics Platform</h3>
        <p>
        DataInsight Pro is a powerful Streamlit-based application designed to help data professionals
        quickly explore, analyze, and visualize their datasets with minimal effort.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Smart Data Loading**: Supports CSV & Excel files
        - **Automated Cleaning**: Handles missing values intelligently
        - **Advanced Visualizations**: Interactive Plotly charts
        - **Time Series Analysis**: Seasonality and decomposition
        """)
    
    with col2:
        st.markdown("""
        - **Correlation Analysis**: Heatmaps and matrix plots
        - **Multivariate Analysis**: Explore complex relationships
        - **Dark/Light Mode**: Customizable interface
        - **Export Capabilities**: Download results easily
        """)
    
    st.subheader("Technology Stack")
    st.markdown("""
    - **Python**: Core programming language
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation and analysis
    - **Plotly**: Interactive visualizations
    - **Seaborn**: Statistical visualizations
    """)
    
    st.subheader("Contact")
    st.markdown("""
    For support or feature requests, please contact:
    - Email: support@datainsightpro.com
    - GitHub: github.com/datainsightpro
    """)

def help_page():
    """Help and documentation page"""
    st.title("❓ Help & Documentation")
    
    st.markdown("""
    <div class="custom-card">
        <h3>Getting Started with DataInsight Pro</h3>
        <p>
        This guide will help you navigate through the application's features and capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("How to upload data"):
        st.markdown("""
        1. Navigate to the **Home** page
        2. Click on the upload area or drag & drop your file
        3. Supported formats: CSV, XLS, XLSX
        4. Wait for the data validation to complete
        """)
    
    with st.expander("Understanding the Summary Page"):
        st.markdown("""
        - **Data Preview**: First 5 rows of your dataset
        - **Column Information**: Data types and uniqueness
        - **Descriptive Stats**: Statistical overview
        - **Missing Values**: Data quality indicators
        """)
    
    with st.expander("Visualization Guide"):
        st.markdown("""
        - **Distribution Analysis**: Histograms, box plots, violin plots
        - **Correlation**: Heatmap of relationships
        - **Time Series**: Trend, seasonality, decomposition
        - **Multivariate**: Scatter, bubble, parallel coordinates
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        - **File not loading**: Check format and encoding
        - **Visuals not showing**: Ensure numeric columns are selected
        - **Slow performance**: Try with smaller datasets first
        - **Error messages**: Note the details and contact support
        """)

# ================================================
# MAIN APP
# ================================================

def main():
    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = None
    if 'clean_data' not in st.session_state:
        st.session_state['clean_data'] = None
    
    # Dark/light mode toggle
    with st.sidebar:
        st.title("Settings")
        dark_mode = st.toggle("Dark Mode", value=False)
        if dark_mode:
            st.markdown("<style>[data-theme='light'] {display: none;}</style>", unsafe_allow_html=True)
            st.markdown("<style>[data-theme='dark'] {display: block;}</style>", unsafe_allow_html=True)
        else:
            st.markdown("<style>[data-theme='dark'] {display: none;}</style>", unsafe_allow_html=True)
            st.markdown("<style>[data-theme='light'] {display: block;}</style>", unsafe_allow_html=True)
    
    # Navigation menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Summary", "Visuals", "About", "Help"],
            icons=["house", "clipboard-data", "bar-chart-line", "info-circle", "question-circle"],
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
            }
        )
    
    # Page routing
    if selected == "Home":
        home_page()
    elif selected == "Summary":
        summary_page()
    elif selected == "Visuals":
        visuals_page()
    elif selected == "About":
        about_page()
    elif selected == "Help":
        help_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <small>DataInsight Pro v1.0</small><br>
        <small>© 2023 Analytics Solutions</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Export options
    if 'clean_data' in st.session_state and st.session_state['clean_data'] is not None:
        with st.sidebar:
            st.markdown("---")
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download CSV",
                    data=st.session_state['clean_data'].to_csv(index=False).encode('utf-8'),
                    file_name='cleaned_data.csv',
                    mime='text/csv'
                )
            with col2:
                excel_data = get_excel_download_link(st.session_state['clean_data'])
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name='cleaned_data.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

if __name__ == "__main__":
    main()