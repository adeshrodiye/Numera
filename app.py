import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import io
import os
import base64
from datetime import datetime
import tempfile
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import requests
from streamlit_lottie import st_lottie
from io import StringIO
import json
import re
from streamlit_option_menu import option_menu
import openpyxl

# Set page config
st.set_page_config(
    page_title="Numera | Data Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
APP_VERSION = "1.0.0"

# Initialize session state

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False
if 'insights_generated' not in st.session_state:
    st.session_state.insights_generated = False
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'numerical_columns' not in st.session_state:
    st.session_state.numerical_columns = []

# Apply dark mode theme if enabled
def apply_theme():
    """Apply the selected theme (light/dark) to the app"""
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;  #represent dark background
                color: #ffffff; #represent white text
            }
            .sidebar .sidebar-content {
                background-color: #262730; #represent dark sidebar
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #ffffff;
                color: #31333F;
            }
            </style>
        """, unsafe_allow_html=True)

# Apply the current theme
apply_theme()

# Custom CSS for styling
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        margin-bottom: 1rem;
    }
    .logo-title {
        display: flex;
        align-items: center;
    }
    .logo {
        width: 50px;
        margin-right: 10px;
    }
    .title {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .subtitle {
        font-size: 1rem;
        opacity: 0.7;
        margin: 0;
    }
    .insight-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .insight-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def load_lottie_animation(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


def get_placeholder_logo():
    """Return a base64 placeholder logo if GitHub logo can't be fetched"""
    # Create a simple placeholder with Matplotlib
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.text(0.5, 0.5, 'N', ha='center', va='center', fontsize=50, color='#1f77b4')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return f"data:image/png;base64,{img_str}"

def display_header():
    """Display the application header with logo and title"""
    logo_url = get_placeholder_logo()
    
    header_html = f"""
    <div class="header-container">
        <div class="logo-title">
            <img src="{logo_url}" class="logo" alt="Numera Logo">
            <div>
                <h1 class="title">Numera</h1>
                <p class="subtitle">Advanced Data Analytics Platform</p>
            </div>
        </div>
        <div class="version">v{APP_VERSION}</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def toggle_dark_mode():
    """Toggle between light and dark mode"""
    st.session_state.dark_mode = not st.session_state.dark_mode
    apply_theme()


def load_data(file):
    """Load data from uploaded file (CSV or Excel)"""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.csv':
            # Try different encodings and delimiters for CSV
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='latin1')
                except:
                    df = pd.read_csv(file, encoding='utf-8-sig')
            except:
                # Try different delimiters
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(file, delimiter=delimiter)
                        if len(df.columns) > 1:  # Successful parsing should have multiple columns
                            break
                    except:
                        continue
        
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Store the data and filename in session state
        st.session_state.data = df
        st.session_state.filename = file.name
        st.session_state.summary_generated = False
        st.session_state.insights_generated = False
        
        # Identify column types
        identify_column_types(df)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def identify_column_types(df):
    """Identify categorical and numerical columns in the dataframe"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Also check if numeric columns with few unique values should be treated as categorical
    for col in numerical_cols.copy():
        if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
            categorical_cols.append(col)
            numerical_cols.remove(col)
    
    # Date columns detection
    date_cols = []
    for col in df.columns:
        if col not in numerical_cols and col not in categorical_cols:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass
    
    st.session_state.numerical_columns = numerical_cols
    st.session_state.categorical_columns = categorical_cols + date_cols

def generate_data_summary(df):
    """Generate a comprehensive summary of the dataset"""
    if df is None:
        return None
    
    # Basic dataset information
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    dtypes = df.dtypes.value_counts().to_dict()
    dtypes_str = ", ".join([f"{v} {k}" for k, v in dtypes.items()])
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_pct = round(missing_values / rows * 100,2) #changed round syntax
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    duplicate_pct = round(duplicates / rows * 100, 2) #changed round syntax
    
    # Summary statistics
    numeric_summary = df.describe().T if not df.select_dtypes(include=['int64', 'float64']).empty else None
    
    # Categorical column statistics
    cat_summary = {}
    for col in st.session_state.categorical_columns:
        value_counts = df[col].value_counts().head(5).to_dict()
        top_values = ", ".join([f"{k} ({v})" for k, v in value_counts.items()])
        unique_count = df[col].nunique()
        unique_pct = round(unique_count / rows * 100,2)
        cat_summary[col] = {
            "unique_values": unique_count,
            "unique_pct": unique_pct,
            "top_values": top_values,
            "missing": missing_values[col],
            "missing_pct": missing_pct[col]
        }
    
    # Create a summary dictionary
    summary = {
        "basic_info": {
            "rows": rows,
            "columns": cols,
            "memory_usage_mb": round(memory_usage, 2),
            "dtypes": dtypes_str
        },
        "data_quality": {
            "missing_values_total": missing_values.sum(),
            "missing_values_pct": round(missing_values.sum() / (rows * cols) * 100,2), #changed round syntax
            "columns_with_missing": columns_with_missing,
            "duplicates": duplicates,
            "duplicate_pct": duplicate_pct
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": cat_summary
    }
    
    st.session_state.summary_generated = True
    return summary

def clean_data(df, methods=None):
    """Clean the data based on selected methods"""
    if df is None:
        return None
    
    df_cleaned = df.copy()
    
    if methods is None:
        methods = ["remove_duplicates", "handle_missing_values"]
    
    # Remove duplicates
    if "remove_duplicates" in methods:
        df_cleaned = df_cleaned.drop_duplicates()
    
    # Handle missing values
    if "handle_missing_values" in methods:
        # For numerical columns: impute with median
        num_imputer = SimpleImputer(strategy='median')
        for col in st.session_state.numerical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = num_imputer.fit_transform(df_cleaned[[col]])
        
        # For categorical columns: impute with mode/most frequent
        cat_imputer = SimpleImputer(strategy='most_frequent')
        for col in st.session_state.categorical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = cat_imputer.fit_transform(df_cleaned[[col]])
    
    return df_cleaned

def generate_insights(df):
    """Generate key insights from the data"""
    if df is None or df.empty:
        return []
    
    insights = []
    
    # 1. Distribution skewness for numerical columns
    for col in st.session_state.numerical_columns[:3]:  # Limit to first 3 numerical columns
        try:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append({
                    "title": f"Distribution of {col} is highly skewed",
                    "description": f"The {col} column has a skewness of {skewness:.2f}, indicating a {direction}-skewed distribution. This may affect statistical analyses.",
                    "viz_type": "histogram",
                    "column": col
                })
        except:
            pass
    
    # 2. Correlation insights
    if len(st.session_state.numerical_columns) >= 2:
        try:
            corr_matrix = df[st.session_state.numerical_columns].corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # Strong correlation
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if corr_pairs:
                top_corr = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[0]
                corr_type = "positive" if top_corr[2] > 0 else "negative"
                insights.append({
                    "title": f"Strong {corr_type} correlation detected",
                    "description": f"There is a strong {corr_type} correlation of {top_corr[2]:.2f} between {top_corr[0]} and {top_corr[1]}.",
                    "viz_type": "scatter",
                    "columns": [top_corr[0], top_corr[1]]
                })
        except:
            pass
    
    # 3. Categorical value distributions
    for col in st.session_state.categorical_columns[:2]:  # Limit to first 2 categorical columns
        try:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1 and len(value_counts) <= 10:
                top_category = value_counts.index[0]
                top_pct = round(value_counts.iloc[0] / value_counts.sum() * 100,2) #changed round syntax
                
                if top_pct > 70:  # Highly imbalanced
                    insights.append({
                        "title": f"Imbalanced distribution in {col}",
                        "description": f"The category '{top_category}' represents {top_pct}% of values in {col}, which may indicate an imbalanced dataset.",
                        "viz_type": "bar",
                        "column": col
                    })
        except:
            pass
    
    # 4. Missing value patterns
    missing_cols = df.columns[df.isnull().mean() > 0].tolist()
    if len(missing_cols) >= 2:
        insights.append({
            "title": "Missing value patterns detected",
            "description": f"There are {len(missing_cols)} columns with missing values. Analyzing patterns might reveal data collection issues.",
            "viz_type": "heatmap",
            "columns": missing_cols
        })
    
    # 5. Outlier detection for numerical columns
    for col in st.session_state.numerical_columns[:2]:  # Limit to first 2 numerical columns
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
            
            if outliers > 0 and (outliers / len(df) * 100) > 1:  # More than 1% are outliers
                insights.append({
                    "title": f"Outliers detected in {col}",
                    "description": f"Found {outliers} outliers ({(outliers/len(df)*100):.2f}%) in {col}, which may affect statistical analyses.",
                    "viz_type": "box",
                    "column": col
                })
        except:
            pass
    
    st.session_state.insights_generated = True
    return insights[:5]  # Limit to top 5 insights

def create_visualization(df, viz_type, columns, **kwargs):
    """Create a visualization based on the specified type and columns"""
    if df is None or df.empty:
        return None
    
    try:
        if viz_type == "histogram":
            fig = px.histogram(df, x=columns[0], 
                              nbins=kwargs.get('nbins', 20),
                              marginal=kwargs.get('marginal', 'box'),
                              title=f"Distribution of {columns[0]}",
                              color_discrete_sequence=['#3366cc'])
            
        elif viz_type == "bar":
            # Get value counts and sort if requested
            value_counts = df[columns[0]].value_counts().reset_index()
            value_counts.columns = [columns[0], 'count']
            
            if kwargs.get('sort', True):
                value_counts = value_counts.sort_values('count', ascending=False)
            
            if len(value_counts) > 20:  # Limit to top 20 categories for readability
                value_counts = value_counts.head(20)
                title = f"Top 20 Categories in {columns[0]}"
            else:
                title = f"Categories in {columns[0]}"
                
            fig = px.bar(value_counts, x=columns[0], y='count', 
                        title=title,
                        color_discrete_sequence=['#3366cc'])
            
        elif viz_type == "scatter":
            fig = px.scatter(df, x=columns[0], y=columns[1],
                            title=f"{columns[1]} vs {columns[0]}",
                            opacity=0.7,
                            color=kwargs.get('color', None),
                            size=kwargs.get('size', None),
                            trendline=kwargs.get('trendline', 'ols') if len(df) > 2 else None,
                            color_discrete_sequence=['#3366cc'])
            
        elif viz_type == "line":
            fig = px.line(df, x=columns[0], y=columns[1],
                         title=f"{columns[1]} over {columns[0]}",
                         color=kwargs.get('color', None),
                         markers=kwargs.get('markers', True),
                         color_discrete_sequence=['#3366cc'])
            
        elif viz_type == "box":
            fig = px.box(df, y=columns[0],
                        title=f"Box Plot of {columns[0]}",
                        color=kwargs.get('color', None),
                        notched=kwargs.get('notched', False),
                        color_discrete_sequence=['#3366cc'])
            
        elif viz_type == "heatmap":
            # For heatmap of correlations
            if len(columns) > 1 and all(col in st.session_state.numerical_columns for col in columns):
                corr = df[columns].corr()
                fig = px.imshow(corr,
                               title="Correlation Matrix",
                               color_continuous_scale=kwargs.get('colorscale', 'RdBu_r'),
                               zmin=-1, zmax=1)
            # For heatmap of missing values
            else:
                # Create a binary mask of missing values (1 for missing, 0 for present)
                missing_mask = df[columns].isnull().astype(int)
                # Transpose for better visualization if there are many columns
                if len(columns) > 10:
                    missing_mask = missing_mask.T
                    title = "Missing Value Patterns (Transposed)"
                else:
                    title = "Missing Value Patterns"
                
                fig = px.imshow(missing_mask,
                               title=title,
                               color_continuous_scale=kwargs.get('colorscale', 'Blues'),
                               labels=dict(x="Data Points", y="Variables", color="Missing"))
            
        elif viz_type == "pie":
            value_counts = df[columns[0]].value_counts()
            
            # If too many categories, limit to top ones
            if len(value_counts) > 10:
                other_count = value_counts[10:].sum()
                value_counts = value_counts[:9].append(pd.Series([other_count], index=['Other']))
                
            fig = px.pie(values=value_counts.values, 
                        names=value_counts.index,
                        title=f"Distribution of {columns[0]}",
                        hole=kwargs.get('hole', 0.3))
            
        else:
            st.warning(f"Visualization type '{viz_type}' not supported.")
            return None
        
        # Add grid and improve layout
        fig.update_layout(
            template="plotly_white" if not st.session_state.dark_mode else "plotly_dark",
            xaxis_title=columns[0] if len(columns) > 0 else "",
            yaxis_title=columns[1] if len(columns) > 1 else "Count",
            legend_title="Legend",
            font=dict(family="Arial, sans-serif", size=12),
            autosize=True,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None
    
    

# Main application after login
def display_main_app():
    """Display the main application after login"""
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Use option_menu for better navigation
        selected = option_menu(
            "Numera Analytics",
            ["Home", "Summary", "Visuals", "Insights", "Help"],
            icons=['house', 'clipboard-data', 'graph-up', 'lightbulb', 'question-circle'],
            menu_icon="app-indicator",
            default_index=0,
        )
        
        st.markdown("---")
        
        # File uploader
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.success(f"Loaded {st.session_state.filename} successfully!")
                        st.dataframe(df.head(5))
        
        # Settings section
        st.markdown("---")
        st.subheader("Settings")
        
        # Theme toggle
        theme_col1, theme_col2 = st.columns([3, 1])
        with theme_col1:
            st.write("Theme:")
        with theme_col2:
            if st.button("🌓" if st.session_state.dark_mode else "☀️"):
                toggle_dark_mode()
                st.experimental_rerun()
        
        
        
        # App info
        st.markdown("---")
        st.markdown(f"<small>Numera v{APP_VERSION}</small>", unsafe_allow_html=True)
        st.markdown("<small>© 2025 Numera Analytics</small>", unsafe_allow_html=True)
    
    # Main content area based on selected tab
    if selected == "Home":
        display_home_tab()
    elif selected == "Summary":
        display_summary_tab()
    elif selected == "Visuals":
        display_visuals_tab()
    elif selected == "Insights":
        display_insights_tab()
    elif selected == "Help":
        display_help_tab()

def display_home_tab():
    """Display the home tab content"""
    display_header()
    
    st.markdown("## Welcome to Numera Analytics Platform")
    st.markdown("Numera helps you unlock insights from your data through automatic exploration, cleaning, and visualization.")
    
    # Welcome animation
    welcome_animation = load_lottie_animation('https://assets3.lottiefiles.com/packages/lf20_khzniaya.json')
    if welcome_animation:
            st_lottie(welcome_animation, height=300, key="welcome_animation")
        
    st.markdown("""
        ### Get Started
        1. Upload your CSV or Excel file using the sidebar uploader
        2. Click "Load Data" to process your file
        3. Explore the different tabs to analyze your data
        
        ### Features
        - **Summary**: Get comprehensive statistics and data quality metrics
        - **Visuals**: Create interactive visualizations with your data
        - **Insights**: Discover key patterns and relationships automatically
        - **Help**: Learn how to use Numera effectively
        """)

def display_summary_tab():
    """Display the summary tab content"""
    display_header()
    
    st.markdown("## Data Summary")
    
    if st.session_state.data is None:
        st.info("Please upload a file first using the sidebar.")
        return
    
    df = st.session_state.data
    
    # Generate summary if not already done
    if not st.session_state.summary_generated:
        with st.spinner("Generating summary..."):
            summary = generate_data_summary(df)
    else:
        summary = generate_data_summary(df)
    
    if summary:
        # Create tabs for different summary sections
        basic_tab, quality_tab, numerical_tab, categorical_tab = st.tabs(["Basic Info", "Data Quality", "Numerical Columns", "Categorical Columns"])
        
        with basic_tab:
            st.subheader("Dataset Overview")

            basic_info = summary["basic_info"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{basic_info['rows']:,}")
            with col2:
                st.metric("Columns", basic_info['columns'])
            with col3:
                st.metric("Memory Usage", f"{basic_info['memory_usage_mb']} MB")
            with col4:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            st.markdown(f"**Column Data Types**: {basic_info['dtypes']}")
            
            # Display dataset schema
            st.subheader("Dataset Schema")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': round(df.isnull().sum() / len(df) * 100,2), # changed round syntax
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(schema_df)
        
        with quality_tab:
            st.subheader("Data Quality Metrics")
            quality = summary["data_quality"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Values", f"{quality['missing_values_total']:,}")
                st.metric("Missing Values %", f"{quality['missing_values_pct']}%")
            with col2:
                st.metric("Duplicate Rows", f"{quality['duplicates']:,}")
                st.metric("Duplicate Rows %", f"{quality['duplicate_pct']}%")
            
            if quality['columns_with_missing']:
                st.subheader("Columns with Missing Values")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum(),
                    'Missing %': round(df.isnull().sum() / len(df) * 100,2) #changed round syntax
                }).sort_values('Missing Count', ascending=False)
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                st.dataframe(missing_df)
                
                # Visualization of missing values
                if len(quality['columns_with_missing']) > 0:
                    st.subheader("Missing Values Visualization")
                    missing_cols = df.columns[df.isnull().sum() > 0]
                    if len(missing_cols) > 0:
                        fig = create_visualization(df, "heatmap", missing_cols)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset.")
            
            if quality['duplicates'] > 0:
                st.subheader("Duplicate Rows")
                if st.button("Show Duplicate Rows"):
                    st.dataframe(df[df.duplicated(keep=False)].head(100))
            else:
                st.success("No duplicate rows found in the dataset.")
        
        with numerical_tab:
            if summary["numeric_summary"] is not None and not summary["numeric_summary"].empty:
                st.subheader("Numerical Column Statistics")
                num_summary = summary["numeric_summary"]
                
                # Allow user to select columns
                selected_num_cols = st.multiselect(
                    "Select numerical columns to view",
                    options=st.session_state.numerical_columns,
                    default=st.session_state.numerical_columns[:min(5, len(st.session_state.numerical_columns))]
                )
                
                if selected_num_cols:
                    # Display stats for selected columns
                    stats_df = num_summary.loc[selected_num_cols]
                    st.dataframe(stats_df)
                    
                    # Distribution visualizations
                    st.subheader("Distributions")
                    for col in selected_num_cols:
                        st.write(f"**{col}**")
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            hist_fig = create_visualization(df, "histogram", [col])
                            if hist_fig:
                                st.plotly_chart(hist_fig, use_container_width=True)
                        
                        with viz_col2:
                            box_fig = create_visualization(df, "box", [col])
                            if box_fig:
                                st.plotly_chart(box_fig, use_container_width=True)
            else:
                st.info("No numerical columns found in the dataset.")
        
        with categorical_tab:
            if summary["categorical_summary"]:
                st.subheader("Categorical Column Statistics")
                
                # Allow user to select columns
                selected_cat_cols = st.multiselect(
                    "Select categorical columns to view",
                    options=list(summary["categorical_summary"].keys()),
                    default=list(summary["categorical_summary"].keys())[:min(3, len(summary["categorical_summary"]))]
                )
                
                if selected_cat_cols:
                    for col in selected_cat_cols:
                        cat_info = summary["categorical_summary"][col]
                        
                        st.write(f"### {col}")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Unique Values", cat_info["unique_values"])
                        with col2:
                            st.metric("Unique %", f"{cat_info['unique_pct']}%")
                        with col3:
                            st.metric("Missing Values", cat_info["missing"])
                        with col4:
                            st.metric("Missing %", f"{cat_info['missing_pct']}%")
                        
                        st.write(f"**Top Values**: {cat_info['top_values']}")
                        
                        # Bar chart for this categorical column
                        bar_fig = create_visualization(df, "bar", [col])
                        if bar_fig:
                            st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.info("No categorical columns found in the dataset.")
        
        # Data cleaning options
        st.markdown("---")
        st.subheader("Data Cleaning Options")
        
        cleaning_col1, cleaning_col2 = st.columns(2)
        
        with cleaning_col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        
        with cleaning_col2:
            handle_missing = st.checkbox("Handle missing values", value=True)
        
        if st.button("Clean Data"):
            cleaning_methods = []
            if remove_duplicates:
                cleaning_methods.append("remove_duplicates")
            if handle_missing:
                cleaning_methods.append("handle_missing_values")
            
            with st.spinner("Cleaning data..."):
                cleaned_df = clean_data(df, cleaning_methods)
                if cleaned_df is not None:
                    # Compare before/after
                    st.success(f"Data cleaned successfully! Rows before: {len(df)}, Rows after: {len(cleaned_df)}")
                    
                    if len(df) != len(cleaned_df) or df.isnull().sum().sum() != cleaned_df.isnull().sum().sum():
                        comparison_col1, comparison_col2 = st.columns(2)
                        
                        with comparison_col1:
                            st.write("Original Data Preview")
                            st.dataframe(df.head())
                        
                        with comparison_col2:
                            st.write("Cleaned Data Preview")
                            st.dataframe(cleaned_df.head())
                        
                        # Option to replace original data with cleaned data
                        if st.button("Use Cleaned Data"):
                            st.session_state.data = cleaned_df
                            st.session_state.summary_generated = False  # Reset summary
                            st.success("Original data replaced with cleaned data!")
                            st.experimental_rerun()
                    else:
                        st.info("No changes were made during cleaning.")

def display_visuals_tab():
    """Display the visualizations tab content"""
    display_header()
    
    st.markdown("## Data Visualizations")
    
    if st.session_state.data is None:
        st.info("Please upload a file first using the sidebar.")
        return
    
    df = st.session_state.data
    
    # Visualization controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Controls")
    
    viz_types = {
        "Histogram": "histogram", 
        "Bar Chart": "bar", 
        "Scatter Plot": "scatter", 
        "Line Chart": "line", 
        "Box Plot": "box", 
        "Pie Chart": "pie", 
        "Heatmap": "heatmap"
    }
    
    viz_type = st.sidebar.selectbox("Visualization Type", list(viz_types.keys()))
    
    # Column selection based on visualization type
    if viz_types[viz_type] in ["histogram", "bar", "box", "pie"]:
        columns = st.sidebar.selectbox(
            "Select Column",
            options=df.columns.tolist(),
            key=f"{viz_types[viz_type]}_column"
        )
        selected_columns = [columns]
    
    elif viz_types[viz_type] in ["scatter", "line"]:
        col1 = st.sidebar.selectbox(
            "X-Axis",
            options=df.columns.tolist(),
            key=f"{viz_types[viz_type]}_x"
        )
        
        col2 = st.sidebar.selectbox(
            "Y-Axis",
            options=[c for c in df.columns if c != col1],
            key=f"{viz_types[viz_type]}_y"
        )
        
        color_by = st.sidebar.selectbox(
            "Color By (Optional)",
            options=["None"] + [c for c in df.columns if c not in [col1, col2] and df[c].nunique() <= 10],
            key=f"{viz_types[viz_type]}_color"
        )
        
        selected_columns = [col1, col2]
        extra_args = {}
        
        if color_by != "None":
            extra_args["color"] = color_by
        
        if viz_types[viz_type] == "scatter":
            # Add trendline option
            add_trendline = st.sidebar.checkbox("Add Trendline", value=False)
            if add_trendline:
                extra_args["trendline"] = "ols"
            
            # Add size option
            size_by = st.sidebar.selectbox(
                "Size By (Optional)",
                options=["None"] + st.session_state.numerical_columns,
                key="scatter_size"
            )
            
            if size_by != "None":
                extra_args["size"] = size_by
    
    elif viz_types[viz_type] == "heatmap":
        if len(st.session_state.numerical_columns) > 1:
            corr_option = st.sidebar.radio(
                "Heatmap Type",
                ["Correlation Matrix", "Missing Values Pattern"]
            )
            
            if corr_option == "Correlation Matrix":
                # Let user select numerical columns for correlation
                selected_columns = st.sidebar.multiselect(
                    "Select Columns for Correlation",
                    options=st.session_state.numerical_columns,
                    default=st.session_state.numerical_columns[:min(5, len(st.session_state.numerical_columns))]
                )
                
                # Colorscale selection
                colorscale = st.sidebar.selectbox(
                    "Color Scale",
                    options=["RdBu_r", "Viridis", "Plasma", "Blues", "Reds"],
                    index=0
                )
                
                extra_args = {"colorscale": colorscale}
            else:
                # Missing values pattern
                selected_columns = st.sidebar.multiselect(
                    "Select Columns",
                    options=df.columns.tolist(),
                    default=[col for col in df.columns if df[col].isnull().sum() > 0][:10]
                )
                
                colorscale = st.sidebar.selectbox(
                    "Color Scale",
                    options=["Blues", "Reds", "Greens", "Purples"],
                    index=0
                )
                
                extra_args = {"colorscale": colorscale}
        else:
            st.warning("Not enough numerical columns for correlation heatmap.")
            selected_columns = []
            extra_args = {}
    else:
        selected_columns = []
        extra_args = {}
    
    # Additional customization options based on visualization type
    st.sidebar.markdown("---")
    st.sidebar.subheader("Customization")
    
    if viz_types[viz_type] == "histogram":
        nbins = st.sidebar.slider("Number of Bins", min_value=5, max_value=100, value=20)
        marginal = st.sidebar.selectbox(
            "Marginal Plot",
            options=["None", "box", "violin", "rug"],
            index=1
        )
        
        extra_args = {
            "nbins": nbins,
            "marginal": None if marginal == "None" else marginal
        }
    
    elif viz_types[viz_type] == "bar":
        sort_bars = st.sidebar.checkbox("Sort Bars", value=True)
        extra_args = {"sort": sort_bars}
    
    elif viz_types[viz_type] == "box":
        notched = st.sidebar.checkbox("Notched Box Plot", value=False)
        extra_args = {"notched": notched}
    
    elif viz_types[viz_type] == "pie":
        donut = st.sidebar.checkbox("Donut Chart", value=True)
        hole_size = st.sidebar.slider("Hole Size", min_value=0.0, max_value=0.8, value=0.4) if donut else 0.0
        extra_args = {"hole": hole_size}
    
    # Create the visualization
    st.subheader(f"{viz_type} Visualization")
    
    if selected_columns:
        with st.spinner("Creating visualization..."):
            fig = create_visualization(df, viz_types[viz_type], selected_columns, **extra_args)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.download_button(
                    label="Download as HTML",
                    data=io.StringIO(fig.to_html()).getvalue(),
                    file_name=f"{viz_type.lower().replace(' ', '_')}_viz.html",
                    mime="text/html"
                )
            else:
                st.error("Failed to create visualization. Please try different columns or settings.")
    else:
        st.info("Please select columns for visualization.")
    
    # Quick visualization gallery
    st.markdown("---")
    st.subheader("Quick Visualization Gallery")
    st.write("Click to generate common visualizations:")
    
    gallery_col1, gallery_col2, gallery_col3 = st.columns(3)
    
    with gallery_col1:
        if st.button("📊 Data Type Distribution"):
            with st.spinner("Creating visualization..."):
                # Count columns by data type
                dtype_counts = df.dtypes.value_counts().reset_index()
                dtype_counts.columns = ['Data Type', 'Count']
                
                fig = px.bar(dtype_counts, x='Data Type', y='Count', 
                            title="Column Count by Data Type",
                            color_discrete_sequence=['#3366cc'])
                
                st.plotly_chart(fig, use_container_width=True)
    
    with gallery_col2:
        if st.button("🧩 Missing Values Overview"):
            with st.spinner("Creating visualization..."):
                # Calculate missing value percentages
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing %': round(df.isnull().sum() / len(df) * 100,2) #changed round syntax
                }).sort_values('Missing %', ascending=False)
                
                fig = px.bar(missing_df, x='Column', y='Missing %',
                            title="Missing Values by Column (%)",
                            color='Missing %',
                            color_continuous_scale='Blues')
                
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
    
    with gallery_col3:
        if len(st.session_state.numerical_columns) >= 2:
            if st.button("🔄 Correlation Matrix"):
                with st.spinner("Creating visualization..."):
                    # Select only numerical columns with less than 20 columns for readability
                    num_cols = st.session_state.numerical_columns[:min(15, len(st.session_state.numerical_columns))]
                    corr = df[num_cols].corr()
                    
                    fig = px.imshow(corr,
                                   title="Correlation Matrix",
                                   color_continuous_scale='RdBu_r',
                                   zmin=-1, zmax=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.button("🔄 Correlation Matrix", disabled=True)
            st.caption("Need at least 2 numerical columns")

def display_insights_tab():
    """Display the insights tab content"""
    display_header()
    
    st.markdown("## Data Insights")
    
    if st.session_state.data is None:
        st.info("Please upload a file first using the sidebar.")
        return
    
    df = st.session_state.data
    
    # Generate insights if not already done
    if not st.session_state.insights_generated:
        with st.spinner("Discovering insights..."):
            insights = generate_insights(df)
    else:
        insights = generate_insights(df)
    
    if insights:
        st.success(f"Found {len(insights)} key insights in your data!")
        
        # Display each insight with its visualization
        for i, insight in enumerate(insights):
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">📊 {insight['title']}</div>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create visualization based on insight type
                viz_type = insight['viz_type']
                
                if 'column' in insight:
                    columns = [insight['column']]
                elif 'columns' in insight:
                    columns = insight['columns']
                else:
                    columns = []
                
                if columns:
                    fig = create_visualization(df, viz_type, columns)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
    else:
        st.info("No significant insights were found in your data. Try uploading a different dataset or adding more columns.")
    



def display_help_tab():
    """Display the help tab content"""
    display_header()
    
    st.markdown("## Numera Help Center")
    
    help_tabs = st.tabs(["Getting Started"])
    
    with help_tabs[0]:
        st.markdown("""
        ### Getting Started with Numera
        
        Numera is an advanced data analytics platform that helps you explore, visualize, and gain insights from your data automatically.
        
        #### How to use Numera:
        
        1. **Upload Data**: Use the sidebar uploader to upload your CSV or Excel file.
        2. **Load Data**: Click the "Load Data" button to process your file.
        3. **Explore**: Navigate through the different tabs to analyze your data:
           - **Summary**: View comprehensive statistics and data quality metrics
           - **Visuals**: Create custom interactive visualizations
           - **Insights**: Discover key patterns and relationships automatically
        
        #### Supported File Formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        
        #### Data Size Limitations:
        - Recommended maximum: 1,000,000 rows and 100 columns
        - Larger files may be processed but with longer loading times
        """)
    
    
    
# Main application logic
def main():
    """Main application entry point"""
    # Apply theme
    apply_theme()
    
    # Show login screen or main app based on authentication status
    display_main_app()

if __name__ == "__main__":
    main()

            