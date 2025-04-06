
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
from datetime import datetime
import time
import streamlit.components.v1 as components
import os
import re
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Numera Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a session state 

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'users' not in st.session_state:
    st.session_state.users = {'admin': 'password', 'demo': 'demo'}

# Custom CSS for light/dark theme
def apply_theme():
    if st.session_state.dark_theme:
        # Dark theme
        st.markdown("""
        <style>
        .main {background-color: #0E1117; color: #FAFAFA;}
        .stApp {background-color: #0E1117;}
        .css-1d391kg {background-color: #1A1D24;}
        .st-bq {background-color: #262730;}
        .css-1adrfps {background-color: #262730; color: #FAFAFA;}
        .css-1v3fvcr {background-color: #0E1117;}
        .css-1vq4p4l {background-color: #262730;}
        .css-1djdyxw {color: #FAFAFA;}
        .st-cp {background-color: #37393E;}
        .css-18e3th9 {padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem; background-color: #0E1117;}
        h1, h2, h3, h4, h5, h6 {color: #FAFAFA;}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
        .main {background-color: #FFFFFF; color: #000000;}
        .stApp {background-color: #FFFFFF;}
        .css-1d391kg {background-color: #F0F2F6;}
        .st-bq {background-color: #F0F2F6;}
        .css-1adrfps {background-color: #F0F2F6; color: #000000;}
        .css-1v3fvcr {background-color: #FFFFFF;}
        .css-1vq4p4l {background-color: #F0F2F6;}
        .css-1djdyxw {color: #000000;}
        .st-cp {background-color: #FFFFFF;}
        .css-18e3th9 {padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem; background-color: #FFFFFF;}
        h1, h2, h3, h4, h5, h6 {color: #000000;}
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# Generate Numera Logo
import streamlit as st
from PIL import Image
import os

# Function to load the logo from a file on your PC
def create_logo():
    try:
        # Specify the path to your logo file
        logo_path = "E:/Code/Stremlit Insight Application/numera.png"  # file path
        
        # Check if the file exists
        if os.path.exists(logo_path):
            # Open and return the image
            return Image.open(logo_path)
        else:
            st.warning(f"Logo file not found at: {logo_path}")
            return None
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        return None

# Login screen
def render_login():
    initialize_session_state()  # Initialize session state variables
    
    st.title("Welcome to Numera Analytics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        logo = create_logo()
        if logo is not None:
            st.image(logo, width=150)
        else:
            st.write("Numera Analytics")
        
    with col2:
        st.markdown("""
        ### Turn Your Data Into Insights
        Numera is a powerful data analytics platform that helps you explore, 
        analyze and visualize your data with ease.
        """)
    
    st.divider()
    
    login_col1, login_col2, login_col3 = st.columns([1, 1, 1])
    
    with login_col2:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
                
        st.caption("Try username: 'demo' and password: 'demo' for a quick demo")

# Sidebar and navigation
def render_sidebar():
    with st.sidebar:
        logo = create_logo()
        st.image(logo, width=75)
        st.title("Numera Analytics")
        
        # Theme Toggle
        theme_col1, theme_col2 = st.columns([1, 3])
        with theme_col1:
            st.write("Theme:")
        with theme_col2:
            if st.toggle("Dark Mode", value=st.session_state.dark_theme):
                st.session_state.dark_theme = True
            else:
                st.session_state.dark_theme = False
            apply_theme()
        
        st.divider()
        
        # Navigation
        tab = st.radio(
            "Navigation",
            ["Home", "Data Summary", "Visualizations", "Help", "Logout"],
            index=0
        )
        
        st.divider()
        
        # File upload section in sidebar
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                st.session_state.filename = uploaded_file.name
                
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Show data info
                st.write(f"Rows: {st.session_state.df.shape[0]}")
                st.write(f"Columns: {st.session_state.df.shape[1]}")
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        return tab

# Home page
def render_home():
    st.title("Numera Analytics Dashboard")
    
    if st.session_state.df is None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            logo = create_logo()
            st.image(logo, width=200)
            
        with col2:
            st.markdown("""
            ## Welcome to Numera!
            ### Start by uploading a dataset
            
            Use the sidebar to upload your CSV or Excel file to begin exploring your data.
            Numera will help you:
            - Understand your data structure
            - Clean and process your data
            - Create interactive visualizations
            - Discover meaningful insights
            """)
            
        st.divider()
        
        st.subheader("How to use Numera")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1. Upload Data
            Upload your CSV or Excel file using the sidebar uploader.
            """)
            
        with col2:
            st.markdown("""
            #### 2. Explore & Analyze
            Navigate to the Data Summary tab to understand your data.
            """)
            
        with col3:
            st.markdown("""
            #### 3. Visualize
            Create interactive visualizations tailored to your data.
            """)
    
    else:
        st.header(f"Analyzing: {st.session_state.filename}")
        
        # Display quick stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{st.session_state.df.shape[0]:,}")
            
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
            
        with col3:
            missing_percentage = (st.session_state.df.isna().sum().sum() / (st.session_state.df.shape[0] * st.session_state.df.shape[1]) * 100)
            st.metric("Missing Values", f"{missing_percentage:.2f}%")
            
        with col4:
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
            st.metric("Numeric/Categorical", f"{len(numeric_cols)}/{len(categorical_cols)}")
        
        st.divider()
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        st.divider()
        
        # Quick insights
        st.subheader("Quick Insights")
        
        insight_tab1, insight_tab2 = st.tabs(["Data Distribution", "Missing Values"])
        
        with insight_tab1:
            # Show distribution of numeric columns
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select column to visualize:", numeric_cols)
                
                fig = px.histogram(st.session_state.df, x=selected_col, 
                                  title=f"Distribution of {selected_col}",
                                  color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found in the dataset.")
        
        with insight_tab2:
            # Missing values heatmap
            if st.session_state.df.isna().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(st.session_state.df.isna(), yticklabels=False, cmap='viridis', cbar=False)
                plt.title('Missing Values Heatmap')
                st.pyplot(fig)
            else:
                st.success("No missing values in the dataset!")

# Data Summary page
def render_data_summary():
    st.title("Data Summary")
    
    if st.session_state.df is None:
        st.info("Please upload a dataset from the sidebar to view the summary.")
        return
    
    # Tabs for different summary views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistical Summary", "Column Analysis", "Data Quality"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Basic Information:")
            st.write(f"- Rows: {st.session_state.df.shape[0]:,}")
            st.write(f"- Columns: {st.session_state.df.shape[1]}")
            st.write(f"- Memory Usage: {st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Duplicate check
            duplicates = st.session_state.df.duplicated().sum()
            if duplicates > 0:
                st.warning(f"Dataset contains {duplicates} duplicate rows ({duplicates/st.session_state.df.shape[0]:.2%})")
            else:
                st.success("No duplicate rows found")
            
        with col2:
            st.write("Column Types:")
            
            # Count dtypes
            dtype_counts = st.session_state.df.dtypes.value_counts()
            fig = px.pie(names=dtype_counts.index.astype(str), 
                         values=dtype_counts.values, 
                         title="Column Types Distribution",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample Data
        st.subheader("Sample Data")
        sample_size = st.slider("Sample size", min_value=5, max_value=min(100, st.session_state.df.shape[0]), value=10)
        st.dataframe(st.session_state.df.sample(sample_size), use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Summary")
        
        # Select numerical or categorical
        summary_type = st.radio("Select summary type:", ["Numerical", "Categorical"], horizontal=True)
        
        if summary_type == "Numerical":
            numeric_df = st.session_state.df.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                st.write("Numerical Columns Summary:")
                st.dataframe(numeric_df.describe(), use_container_width=True)
                
                # Correlation heatmap
                if numeric_df.shape[1] > 1:
                    st.subheader("Correlation Matrix")
                    
                    # Calculate correlation
                    corr = numeric_df.corr()
                    
                    # Create heatmap
                    fig = px.imshow(corr, 
                                   text_auto=True, 
                                   color_continuous_scale='RdBu_r',
                                   title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical columns found in the dataset.")
        else:
            categorical_df = st.session_state.df.select_dtypes(exclude=['number'])
            
            if not categorical_df.empty:
                st.write("Categorical Columns Summary:")
                
                # Choose column
                selected_col = st.selectbox("Select categorical column:", categorical_df.columns)
                
                # Value counts
                value_counts = categorical_df[selected_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(categorical_df) * 100).round(2)
                    }), use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig = px.bar(x=value_counts.index, 
                                y=value_counts.values,
                                title=f"Distribution of {selected_col}",
                                labels={'x': selected_col, 'y': 'Count'},
                                color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found in the dataset.")
    
    with tab3:
        st.subheader("Column Analysis")
        
        # Column selector
        selected_col = st.selectbox("Select column to analyze:", st.session_state.df.columns)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Column Information:")
            st.write(f"- Name: {selected_col}")
            st.write(f"- Type: {st.session_state.df[selected_col].dtype}")
            st.write(f"- Unique Values: {st.session_state.df[selected_col].nunique()}")
            st.write(f"- Missing Values: {st.session_state.df[selected_col].isna().sum()} ({st.session_state.df[selected_col].isna().mean():.2%})")
            
            if pd.api.types.is_numeric_dtype(st.session_state.df[selected_col]):
                st.write(f"- Min: {st.session_state.df[selected_col].min()}")
                st.write(f"- Max: {st.session_state.df[selected_col].max()}")
                st.write(f"- Mean: {st.session_state.df[selected_col].mean()}")
                st.write(f"- Median: {st.session_state.df[selected_col].median()}")
                st.write(f"- Std Dev: {st.session_state.df[selected_col].std()}")
        
        with col2:
            # Visualization based on column type
            if pd.api.types.is_numeric_dtype(st.session_state.df[selected_col]):
                # For numeric columns
                viz_type = st.radio("Select visualization:", ["Histogram", "Box Plot"], horizontal=True)
                
                if viz_type == "Histogram":
                    fig = px.histogram(st.session_state.df, x=selected_col, 
                                      title=f"Distribution of {selected_col}",
                                      color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.box(st.session_state.df, y=selected_col, 
                               title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # For categorical columns
                fig = px.bar(st.session_state.df[selected_col].value_counts().reset_index(), 
                            x='index', y=selected_col,
                            title=f"Value Counts for {selected_col}",
                            labels={'index': selected_col, selected_col: 'Count'},
                            color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data Quality Assessment")
        
        # Missing Values Analysis
        st.write("Missing Values by Column:")
        
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isna().sum().values,
            'Missing Percentage': (st.session_state.df.isna().mean() * 100).values.round(2)
        }).sort_values('Missing Count', ascending=False)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.dataframe(missing_df, use_container_width=True)
            
        with col2:
            # Only show columns with missing values
            missing_cols = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_cols.empty:
                fig = px.bar(missing_cols, 
                           x='Column', y='Missing Percentage',
                           title="Missing Values by Column (%)",
                           color='Missing Percentage',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in the dataset!")
        
        # Data Type Distribution
        st.subheader("Data Types")
        
        dtype_df = pd.DataFrame({
            'Type': st.session_state.df.dtypes.astype(str),
        }).reset_index().rename(columns={'index': 'Column', 0: 'Type'})
        
        dtype_counts = dtype_df['Type'].value_counts().reset_index().rename(columns={'index': 'Type', 'Type': 'Count'})
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(dtype_df, use_container_width=True)
            
        with col2:
            fig = px.pie(dtype_counts, names='Type', values='Count', 
                       title="Data Type Distribution",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

# Visualizations page
def render_visualizations():
    st.title("Visualizations")
    
    if st.session_state.df is None:
        st.info("Please upload a dataset from the sidebar to create visualizations.")
        return
    
    # Create tabs for different visualization types
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "Column Selector", "Distribution Plots", "Relationship Plots", "Time Series", "Custom Plots"
    ])
    
    with viz_tab1:
        st.subheader("Build Your Visualization")
        
        st.write("Select columns and chart type to create your visualization:")
        
        # Get column types
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
        all_cols = st.session_state.df.columns.tolist()
        
        # Chart type selector
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot", "Histogram", "Heatmap"]
        )
        
        # Show different options based on chart type
        if chart_type == "Bar Chart":
            x_col = st.selectbox("Select X-axis (categorical):", categorical_cols if categorical_cols else all_cols)
            y_col = st.selectbox("Select Y-axis (numeric):", numeric_cols if numeric_cols else ["Count"])
            
            color_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
            sort_by = st.radio("Sort by:", ["Value", "Alphabetical"], horizontal=True)
            
            if st.button("Generate Bar Chart"):
                st.subheader("Bar Chart")
                
                if y_col == "Count" or y_col not in numeric_cols:
                    # Count plot
                    value_counts = st.session_state.df[x_col].value_counts().reset_index()
                    
                    if sort_by == "Value":
                        value_counts = value_counts.sort_values(x_col, ascending=False)
                    
                    fig = px.bar(value_counts, x='index', y=x_col,
                                title=f"Count of {x_col}",
                                labels={'index': x_col, x_col: 'Count'},
                                color='index' if color_col == "None" else None,
                                color_discrete_sequence=px.colors.qualitative.Plotly)
                else:
                    # Regular bar chart
                    if color_col != "None":
                        fig = px.bar(st.session_state.df, x=x_col, y=y_col, 
                                    color=color_col,
                                    title=f"{y_col} by {x_col} (grouped by {color_col})")
                    else:
                        data = st.session_state.df.groupby(x_col)[y_col].mean().reset_index()
                        
                        if sort_by == "Value":
                            data = data.sort_values(y_col, ascending=False)
                        
                        fig = px.bar(data, x=x_col, y=y_col,
                                    title=f"{y_col} by {x_col}",
                                    color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Line Chart":
            x_col = st.selectbox("Select X-axis:", all_cols)
            y_cols = st.multiselect("Select Y-axis (numeric):", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [])
            
            if st.button("Generate Line Chart"):
                st.subheader("Line Chart")
                
                fig = go.Figure()
                
                for y_col in y_cols:
                    # Sort by x column to make sure line is connected properly
                    sorted_df = st.session_state.df.sort_values(x_col)
                    
                    fig.add_trace(go.Scatter(
                        x=sorted_df[x_col],
                        y=sorted_df[y_col],
                        mode='lines+markers',
                        name=y_col
                    ))
                
                fig.update_layout(
                    title=f"Line Chart: {', '.join(y_cols)} by {x_col}",
                    xaxis_title=x_col,
                    yaxis_title="Value",
                    legend_title="Metrics"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis (numeric):", numeric_cols)
            y_col = st.selectbox("Select Y-axis (numeric):", [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols)
            
            color_col = st.selectbox("Color by (optional):", ["None"] + all_cols)
            size_col = st.selectbox("Size by (optional):", ["None"] + numeric_cols)
            
            if st.button("Generate Scatter Plot"):
                st.subheader("Scatter Plot")
                
                if color_col != "None" and size_col != "None":
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col, 
                                    color=color_col, size=size_col,
                                    title=f"Scatter Plot: {y_col} vs {x_col}",
                                    labels={x_col: x_col, y_col: y_col})
                elif color_col != "None":
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col, 
                                    color=color_col,
                                    title=f"Scatter Plot: {y_col} vs {x_col}",
                                    labels={x_col: x_col, y_col: y_col})
                elif size_col != "None":
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col, 
                                    size=size_col,
                                    title=f"Scatter Plot: {y_col} vs {x_col}",
                                    labels={x_col: x_col, y_col: y_col})
                else:
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col,
                                    title=f"Scatter Plot: {y_col} vs {x_col}",
                                    labels={x_col: x_col, y_col: y_col})
                
                # Add trendline
                add_trendline = st.checkbox("Add trendline")
                if add_trendline:
                    fig.update_traces(mode='markers')
                    
                    # Compute and add trendline
                    x_values = st.session_state.df[x_col].dropna()
                    y_values = st.session_state.df[y_col].dropna()
                    
                    # Create mask for valid values in both columns
                    mask = (~pd.isna(x_values)) & (~pd.isna(y_values))
                    
                    if mask.sum() >= 2:  # Need at least 2 points for a line
                        z = np.polyfit(x_values[mask], y_values[mask], 1)
                        p = np.poly1d(z)
                        
                        x_range = np.linspace(min(x_values), max(x_values), 100)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            name='Trendline',
                            line=dict(color='red', dash='dash')
                        ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation
                if x_col in numeric_cols and y_col in numeric_cols:
                    correlation = st.session_state.df[[x_col, y_col]].corr().iloc[0, 1]
                    st.info(f"Correlation between {x_col} and {y_col}: {correlation:.4f}")
        
        elif chart_type == "Pie Chart":
            cat_col = st.selectbox("Select category column:", categorical_cols if categorical_cols else all_cols)
            value_col = st.selectbox("Select value column (optional):", ["Count"] + numeric_cols)
            
            if st.button("Generate Pie Chart"):
                st.subheader("Pie Chart")
                
                if value_col == "Count":
                    # Simple count for pie chart
                    value_counts = st.session_state.df[cat_col].value_counts()
                    
                    fig = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        title=f"Distribution of {cat_col}"
                    )
                else:
                    # Aggregate by category
                    agg_data = st.session_state.df.groupby(cat_col)[value_col].sum().reset_index()
                    
                    fig = px.pie(
                        agg_data,
                        names=cat_col,
                        values=value_col,
                        title=f"Distribution of {value_col} by {cat_col}"
                    )
                
                # Update layout
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            y_col = st.selectbox("Select value column (numeric):", numeric_cols)
            x_col = st.selectbox("Group by (categorical, optional):", ["None"] + categorical_cols)
            
            if st.button("Generate Box Plot"):
                st.subheader("Box Plot")
                
                if x_col == "None":
                    fig = px.box(st.session_state.df, y=y_col,
                                title=f"Box Plot of {y_col}")
                else:
                    fig = px.box(st.session_state.df, x=x_col, y=y_col,
                                title=f"Box Plot of {y_col} by {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.write("Summary Statistics:")
                st.dataframe(st.session_state.df[y_col].describe().reset_index().rename(columns={"index": "Metric", 0: "Value"}))
        
        elif chart_type == "Histogram":
            col = st.selectbox("Select column (numeric):", numeric_cols)
            bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
            
            if st.button("Generate Histogram"):
                st.subheader("Histogram")
                
                fig = px.histogram(st.session_state.df, x=col, nbins=bins,
                                 title=f"Histogram of {col}",
                                 color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
                
                # Add KDE (Kernel Density Estimate)
                add_kde = st.checkbox("Add density curve")
                if add_kde:
                    df_kde = st.session_state.df[col].dropna()
                    
                    if not df_kde.empty:
                        hist_data = [df_kde]
                        group_labels = [col]
                        
                        # Create distplot with custom bin size
                        fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=bins)
                        fig_kde.update_layout(title_text=f'Histogram with KDE of {col}')
                        st.plotly_chart(fig_kde, use_container_width=True)
                else:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for a heatmap")
            else:
                selected_cols = st.multiselect("Select numeric columns:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
                
                if st.button("Generate Heatmap") and len(selected_cols) >= 2:
                    st.subheader("Correlation Heatmap")
                    
                    # Calculate correlation matrix
                    corr_matrix = st.session_state.df[selected_cols].corr()
                    
                    # Create heatmap
                    fig = px.imshow(corr_matrix,
                                   text_auto=True,
                                   color_continuous_scale='RdBu_r',
                                   title="Correlation Heatmap")
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        st.subheader("Distribution Plots")
        
        dist_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot", "Violin Plot", "Distribution Comparison"])
        
        if dist_type == "Histogram":
            col = st.selectbox("Select column (Distribution):", numeric_cols)
            bins = st.slider("Number of bins (Distribution):", min_value=5, max_value=100, value=20)
            
            fig = px.histogram(st.session_state.df, x=col, nbins=bins,
                             title=f"Histogram of {col}",
                             color_discrete_sequence=['#3366cc' if not st.session_state.dark_theme else '#4287f5'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Mean", f"{st.session_state.df[col].mean():.2f}")
            with stats_col2:
                st.metric("Median", f"{st.session_state.df[col].median():.2f}")
            with stats_col3:
                st.metric("Std Dev", f"{st.session_state.df[col].std():.2f}")
        
        elif dist_type == "Box Plot":
            col = st.selectbox("Select column (Box Plot):", numeric_cols)
            group_by = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
            
            if group_by == "None":
                fig = px.box(st.session_state.df, y=col,
                            title=f"Box Plot of {col}")
            else:
                fig = px.box(st.session_state.df, x=group_by, y=col,
                            title=f"Box Plot of {col} by {group_by}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif dist_type == "Violin Plot":
            col = st.selectbox("Select column (Violin):", numeric_cols)
            group_by = st.selectbox("Group by (optional, Violin):", ["None"] + categorical_cols)
            
            if group_by == "None":
                fig = px.violin(st.session_state.df, y=col,
                               box=True, points="all",
                               title=f"Violin Plot of {col}")
            else:
                fig = px.violin(st.session_state.df, x=group_by, y=col,
                               box=True, points="all",
                               title=f"Violin Plot of {col} by {group_by}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif dist_type == "Distribution Comparison":
            cols = st.multiselect("Select columns to compare:", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            
            if cols:
                # Create a Figure
                fig = go.Figure()
                
                for col in cols:
                    # Add histogram for each column
                    fig.add_trace(go.Histogram(
                        x=st.session_state.df[col],
                        name=col,
                        opacity=0.7,
                        nbinsx=30,
                        histnorm='probability'
                    ))
                
                # Customize layout
                fig.update_layout(
                    title="Distribution Comparison",
                    xaxis_title="Value",
                    yaxis_title="Probability",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        st.subheader("Relationship Plots")
        
        rel_type = st.selectbox("Select relationship plot type:", ["Scatter Plot", "Scatter Matrix", "Bubble Chart", "Heatmap"])
        
        if rel_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis column:", numeric_cols)
            y_col = st.selectbox("Select Y-axis column:", [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols)
            color_col = st.selectbox("Color by (optional, Scatter):", ["None"] + all_cols)
            
            if color_col != "None":
                fig = px.scatter(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                title=f"{y_col} vs {x_col} (colored by {color_col})")
            else:
                fig = px.scatter(st.session_state.df, x=x_col, y=y_col,
                                title=f"{y_col} vs {x_col}")
            
            # Add trendline
            add_trendline = st.checkbox("Add trendline (Scatter)")
            if add_trendline:
                fig.update_traces(mode='markers')
                
                # Filter for valid data points
                valid_data = st.session_state.df[[x_col, y_col]].dropna()
                
                if len(valid_data) >= 2:  # Need at least 2 points for a trendline
                    # Compute linear fit
                    z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
                    p = np.poly1d(z)
                    
                    # Add trendline to plot
                    x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
                    fig.add_trace(go.Scatter(x=x_range, y=p(x_range),
                                          mode='lines', name='Trendline',
                                          line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            correlation = st.session_state.df[[x_col, y_col]].corr().iloc[0, 1]
            st.info(f"Correlation between {x_col} and {y_col}: {correlation:.4f}")
        
        elif rel_type == "Scatter Matrix":
            # Limit to a reasonable number of columns
            max_cols = min(6, len(numeric_cols))
            selected_cols = st.multiselect("Select columns for scatter matrix:", 
                                          numeric_cols, 
                                          default=numeric_cols[:min(4, max_cols)])
            
            color_col = st.selectbox("Color by (optional, Matrix):", ["None"] + categorical_cols)
            
            if selected_cols:
                if color_col != "None":
                    fig = px.scatter_matrix(
                        st.session_state.df,
                        dimensions=selected_cols,
                        color=color_col,
                        title="Scatter Matrix"
                    )
                else:
                    fig = px.scatter_matrix(
                        st.session_state.df,
                        dimensions=selected_cols,
                        title="Scatter Matrix"
                    )
                
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        
        elif rel_type == "Bubble Chart":
            x_col = st.selectbox("Select X-axis (Bubble):", numeric_cols)
            y_col = st.selectbox("Select Y-axis (Bubble):", [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols)
            size_col = st.selectbox("Select Size column:", [col for col in numeric_cols if col != x_col and col != y_col] if len(numeric_cols) > 2 else numeric_cols)
            color_col = st.selectbox("Color by (optional, Bubble):", ["None"] + all_cols)
            
            if color_col != "None":
                fig = px.scatter(st.session_state.df, x=x_col, y=y_col, 
                                size=size_col, color=color_col,
                                title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})")
            else:
                fig = px.scatter(st.session_state.df, x=x_col, y=y_col, 
                                size=size_col,
                                title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif rel_type == "Heatmap":
            # Select columns for heatmap
            selected_cols = st.multiselect("Select columns for heatmap:", 
                                         numeric_cols, 
                                         default=numeric_cols[:min(5, len(numeric_cols))])
            
            if selected_cols and len(selected_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = st.session_state.df[selected_cols].corr()
                
                # Create heatmap
                fig = px.imshow(corr_matrix,
                               text_auto=True,
                               color_continuous_scale='RdBu_r',
                               title="Correlation Heatmap")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                corr_pairs = []
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        corr_pairs.append((selected_cols[i], selected_cols[j], corr_value))
                
                corr_df = pd.DataFrame(corr_pairs, columns=['Column 1', 'Column 2', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                st.write("Strongest Correlations:")
                st.dataframe(corr_df, use_container_width=True)
    
    with viz_tab4:
        st.subheader("Time Series Analysis")
        
        # Detect date columns
        date_cols = []
        for col in st.session_state.df.columns:
            # Check if column name suggests a date
            if any(date_keyword in col.lower() for date_keyword in ['date', 'time', 'day', 'month', 'year']):
                try:
                    # Try to convert to datetime
                    pd.to_datetime(st.session_state.df[col])
                    date_cols.append(col)
                except:
                    pass
            # Check if column type is datetime
            if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]):
                date_cols.append(col)
        
        if not date_cols:
            st.info("No date/time columns detected. To use time series analysis, please select a column to convert to date/time.")
            
            # Let user select a column to convert
            convert_col = st.selectbox("Select column to convert to date/time:", all_cols)
            
            if st.button("Convert to Date/Time"):
                try:
                    st.session_state.df[convert_col + '_datetime'] = pd.to_datetime(st.session_state.df[convert_col])
                    st.success(f"Successfully converted {convert_col} to date/time. Please refresh the time series tab.")
                    date_cols = [convert_col + '_datetime']
                except Exception as e:
                    st.error(f"Error converting to date/time: {e}")
        
        if date_cols:
            date_col = st.selectbox("Select date/time column:", date_cols)
            value_cols = st.multiselect("Select value columns:", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [])
            
            # Resample frequency
            freq_options = {
                "As is": None,
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "M",
                "Quarterly": "Q",
                "Yearly": "Y"
            }
            
            freq = st.selectbox("Resample frequency:", list(freq_options.keys()))
            
            if value_cols:
                # Ensure datetime column
                time_df = st.session_state.df.copy()
                time_df[date_col] = pd.to_datetime(time_df[date_col])
                
                # Sort by date
                time_df = time_df.sort_values(date_col)
                
                # Resample if requested
                if freq_options[freq] is not None:
                    st.write(f"Resampling data to {freq} frequency...")
                    
                    # Set date as index for resampling
                    time_df = time_df.set_index(date_col)
                    
                    # Resample each selected column
                    resampled_dfs = []
                    for col in value_cols:
                        resampled = time_df[[col]].resample(freq_options[freq]).mean()
                        resampled.columns = [f"{col}_{freq}"]
                        resampled_dfs.append(resampled)
                    
                    # Combine resampled dataframes
                    if resampled_dfs:
                        time_df = pd.concat(resampled_dfs, axis=1)
                        time_df = time_df.reset_index()
                        
                        # Update column names for plotting
                        value_cols = [f"{col}_{freq}" for col in value_cols]
                        date_col = time_df.columns[0]
                
                # Create figure
                fig = go.Figure()
                
                for col in value_cols:
                    fig.add_trace(go.Scatter(
                        x=time_df[date_col],
                        y=time_df[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f"Time Series Analysis",
                    xaxis_title=date_col,
                    yaxis_title="Value",
                    legend_title="Metrics"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend analysis
                if st.checkbox("Show trend analysis"):
                    trend_col = st.selectbox("Select column for trend analysis:", value_cols)
                    
                    # Calculate rolling mean
                    window_size = st.slider("Moving average window size:", 2, 30, 7)
                    
                    if len(time_df) >= window_size:
                        # Calculate rolling statistics
                        rolling_mean = time_df[trend_col].rolling(window=window_size).mean()
                        
                        # Plot with rolling mean
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=time_df[date_col],
                            y=time_df[trend_col],
                            mode='lines',
                            name=trend_col
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=time_df[date_col],
                            y=rolling_mean,
                            mode='lines',
                            line=dict(width=3, color='red'),
                            name=f'Rolling Mean ({window_size} periods)'
                        ))
                        
                        fig.update_layout(
                            title=f"Trend Analysis: {trend_col}",
                            xaxis_title=date_col,
                            yaxis_title="Value"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate year-over-year growth if possible
                        if freq_options[freq] in ["D", "W", "M", "Q"]:
                            try:
                                # Calculate percentage change
                                pct_change = time_df[trend_col].pct_change() * 100
                                
                                # Plot percentage change
                                fig = px.bar(
                                    x=time_df[date_col],
                                    y=pct_change,
                                    title=f"Period-over-Period Change (%): {trend_col}",
                                    labels={'x': date_col, 'y': 'Change (%)'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error calculating growth: {e}")
                    else:
                        st.warning(f"Not enough data points for window size {window_size}")
                
                # Seasonality detection
                if st.checkbox("Show seasonality analysis"):
                    season_col = st.selectbox("Select column for seasonality:", value_cols)
                    
                    if len(time_df) >= 12:  # Need enough data for seasonality
                        # Extract month and year if possible
                        try:
                            time_df['month'] = time_df[date_col].dt.month
                            time_df['year'] = time_df[date_col].dt.year
                            
                            # Group by month
                            monthly_avg = time_df.groupby('month')[season_col].mean().reset_index()
                            
                            # Plot monthly pattern
                            fig = px.line(
                                monthly_avg, 
                                x='month', 
                                y=season_col,
                                title=f"Monthly Seasonality Pattern: {season_col}",
                                markers=True
                            )
                            
                            # Set x-axis to show all months
                            fig.update_xaxes(tickvals=list(range(1, 13)),
                                          ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in seasonality analysis: {e}")
                    else:
                        st.warning("Not enough data points for seasonality analysis")
    
    with viz_tab5:
        st.subheader("Custom Plots")
        
        custom_type = st.selectbox("Select custom plot type:", [
            "Advanced Heatmap", "Sunburst Chart", "Parallel Coordinates", "Radar Chart", "3D Scatter"
        ])
        
        if custom_type == "Advanced Heatmap":
            # Get categorical columns for axes
            if len(categorical_cols) < 2:
                st.warning("Need at least 2 categorical columns for a heatmap")
            else:
                x_col = st.selectbox("Select X-axis (categorical):", categorical_cols)
                y_col = st.selectbox("Select Y-axis (categorical):", [col for col in categorical_cols if col != x_col])
                value_col = st.selectbox("Select value column:", ["Count"] + numeric_cols)
                
                # Create cross-tabulation
                if value_col == "Count":
                    # Create count cross-tab
                    crosstab = pd.crosstab(st.session_state.df[y_col], st.session_state.df[x_col])
                else:
                    # Create value-based cross-tab
                    crosstab = pd.pivot_table(
                        st.session_state.df, 
                        values=value_col, 
                        index=y_col, 
                        columns=x_col, 
                        aggfunc='mean'
                    )
                
                # Create heatmap
                fig = px.imshow(
                    crosstab,
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    title=f"Heatmap: {y_col} vs {x_col}" + (f" (Showing: {value_col})" if value_col != "Count" else "")
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif custom_type == "Sunburst Chart":
            if len(categorical_cols) < 2:
                st.warning("Need at least 2 categorical columns for a sunburst chart")
            else:
                # Select hierarchical columns
                path_cols = st.multiselect("Select hierarchy columns (in order):", 
                                         categorical_cols,
                                         default=categorical_cols[:min(3, len(categorical_cols))])
                
                value_col = st.selectbox("Select value column (Sunburst):", ["Count"] + numeric_cols)
                
                if path_cols:
                    if value_col == "Count":
                        # Count-based sunburst
                        fig = px.sunburst(
                            st.session_state.df,
                            path=path_cols,
                            title="Sunburst Chart"
                        )
                    else:
                        # Value-based sunburst
                        fig = px.sunburst(
                            st.session_state.df,
                            path=path_cols,
                            values=value_col,
                            title=f"Sunburst Chart (Values: {value_col})"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif custom_type == "Parallel Coordinates":
            # Select columns for parallel coordinates
            dims = st.multiselect("Select dimensions:", 
                                numeric_cols + categorical_cols,
                                default=(numeric_cols + categorical_cols)[:min(5, len(numeric_cols) + len(categorical_cols))])
            
            color_col = st.selectbox("Color by:", ["None"] + all_cols)
            
            if dims:
                if color_col != "None":
                    fig = px.parallel_coordinates(
                        st.session_state.df,
                        dimensions=dims,
                        color=color_col,
                        title="Parallel Coordinates Plot"
                    )
                else:
                    fig = px.parallel_coordinates(
                        st.session_state.df,
                        dimensions=dims,
                        title="Parallel Coordinates Plot"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif custom_type == "Radar Chart":
            if len(numeric_cols) < 3:
                st.warning("Need at least 3 numeric columns for a radar chart")
            else:
                # Select columns for radar chart
                selected_cols = st.multiselect("Select numeric columns for radar:", 
                                             numeric_cols,
                                             default=numeric_cols[:min(5, len(numeric_cols))])
                
                group_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                
                if selected_cols:
                    if group_col != "None":
                        # Limit number of groups for readability
                        top_groups = st.slider("Number of top groups to display:", 2, 10, 5)
                        
                        # Get top groups by count
                        top_groups_list = st.session_state.df[group_col].value_counts().head(top_groups).index.tolist()
                        
                        # Filter data
                        radar_df = st.session_state.df[st.session_state.df[group_col].isin(top_groups_list)]
                        
                        # Group and aggregate
                        radar_data = radar_df.groupby(group_col)[selected_cols].mean().reset_index()
                        
                        # Create figure
                        fig = go.Figure()
                        
                        for idx, group in enumerate(radar_data[group_col]):
                            values = radar_data.loc[idx, selected_cols].tolist()
                            values.append(values[0])  # Close the loop
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=selected_cols + [selected_cols[0]],  # Close the loop
                                fill='toself',
                                name=str(group)
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                )
                            ),
                            title="Radar Chart"
                        )
                    else:
                        # Single radar chart with average values
                        avg_values = st.session_state.df[selected_cols].mean().tolist()
                        avg_values.append(avg_values[0])  # Close the loop
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=avg_values,
                            theta=selected_cols + [selected_cols[0]],  # Close the loop
                            fill='toself',
                            name='Average'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                )
                            ),
                            title="Radar Chart (Average Values)"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif custom_type == "3D Scatter":
            if len(numeric_cols) < 3:
                st.warning("Need at least 3 numeric columns for a 3D scatter plot")
            else:
                x_col = st.selectbox("Select X-axis (3D):", numeric_cols)
                y_col = st.selectbox("Select Y-axis (3D):", [col for col in numeric_cols if col != x_col])
                z_col = st.selectbox("Select Z-axis (3D):", [col for col in numeric_cols if col != x_col and col != y_col])
                
                color_col = st.selectbox("Color by (3D):", ["None"] + all_cols)
                
                if color_col != "None":
                    fig = px.scatter_3d(
                        st.session_state.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=color_col,
                        title="3D Scatter Plot"
                    )
                else:
                    fig = px.scatter_3d(
                        st.session_state.df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        title="3D Scatter Plot"
                    )
                
                st.plotly_chart(fig, use_container_width=True)

# Help page
def load_help_page():
    st.title("Help & Documentation")
    
    # Add a nice help page header with styling
    st.markdown("""
    <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px'>
        <h2 style='color:#0066cc; margin-bottom:10px'>Welcome to Numera Help Center</h2>
        <p style='font-size:16px'>This guide will help you navigate the Numera Analytics Platform and make the most of its features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create expandable sections for different topics
    with st.expander("🏠 Using the Home Page", expanded=True):
        st.markdown("""
        ### Home Page Overview
        
        The Home page provides an introduction to the Numera Analytics platform and offers quick access to the key features of the application.
        
        #### Features:
        - **Dashboard Summary**: Quick overview of dataset statistics
        - **Quick Navigation**: Easy access to other sections of the app
        - **Dataset Information**: Basic information about the loaded dataset
        
        #### Tips:
        - Bookmark important insights directly from the home page
        - Check the data freshness indicator to ensure you're working with up-to-date information
        """)
    
    with st.expander("📊 Data Summary Section"):
        st.markdown("""
        ### Understanding the Data Summary
        
        The Data Summary section provides detailed information about your dataset, including statistics, data quality metrics, and dataset health.
        
        #### Features:
        - **Data Overview**: Basic statistics about your dataset columns
        - **Missing Values Analysis**: Visualization of missing data patterns
        - **Data Type Distribution**: Breakdown of different data types in your dataset
        - **Correlation Analysis**: Heatmap showing relationships between numerical features
        
        #### Tips:
        - Use the filter options to focus on specific columns or data types
        - Export summary statistics for reporting purposes
        - Pay special attention to data quality metrics to identify potential issues
        """)
    
    with st.expander("📈 Visualizations Guide"):
        st.markdown("""
        ### Creating Effective Visualizations
        
        The Visualizations section allows you to explore your data through interactive charts and graphs.
        
        #### Available Visualizations:
        - **Time Series Analysis**: Track metrics over time periods
        - **Distribution Plots**: Understand the distribution of your data
        - **Relationship Graphs**: Explore correlations and connections
        - **Custom Visualizations**: Create your own charts based on specific requirements
        
        #### Tips:
        - Use the interactive filters to refine your visualizations
        - Save and export visualizations for presentations
        - Try different chart types to find the most effective way to represent your data
        - Use the comparison feature to analyze differences between segments
        """)
    
    with st.expander("🔐 Login & Account Management"):
        st.markdown("""
        ### Managing Your Account
        
        Learn how to manage your Numera account and personalize your experience.
        
        #### Account Features:
        - **User Profiles**: Customize your analytics experience
        - **Saved Insights**: Access your previously saved analyses
        - **Preferences**: Set default views and visualization preferences
        - **Notification Settings**: Configure alerts for important data changes
        
        #### Security Tips:
        - Change your password regularly
        - Enable two-factor authentication for enhanced security
        - Review access logs to monitor account activity
        - Log out when using shared computers
        """)
    
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### Common Questions

        **Q: How do I upload my own dataset?**  
        A: Navigate to the Home page and use the "Upload Dataset" button. The platform supports CSV and Excel files.

        **Q: Can I save my customized visualizations?**  
        A: Yes! Click the "Save" icon in the top-right corner of any visualization to save it to your account.

        **Q: How often is the data refreshed?**  
        A: Data refresh frequency depends on your data source configuration. Check the "Data Freshness" indicator in the top navigation bar.

        **Q: Can I export my insights and visualizations?**  
        A: Yes, all visualizations can be exported as PNG, PDF, or interactive HTML files using the export button.

        **Q: How do I report a bug or request a feature?**  
        A: Use the "Feedback" button in the bottom-right corner of any page to submit your comments to our team.
        """)
    
    # Contact and support section
    st.markdown("---")
    st.subheader("Need Additional Help?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Contact Support
        For technical assistance or questions:
        - 📧 Email: support@numera-analytics.com
        - 🌐 Support Portal: [help.numera-analytics.com](https://help.numera-analytics.com)
        - ☎️ Phone: +1 (800) 555-DATA
        """)
    
    with col2:
        st.markdown("""
        ### Learning Resources
        Improve your skills with our resources:
        - 📚 Documentation: [docs.numera-analytics.com](https://docs.numera-analytics.com)
        - 🎓 Tutorials: [learn.numera-analytics.com](https://learn.numera-analytics.com)
        - 📺 Video Guides: [Numera YouTube Channel](https://youtube.com/numera)
        """)
    
    # Keyboard shortcuts section
    st.markdown("---")
    st.subheader("Keyboard Shortcuts")
    
    shortcuts = {
        "Ctrl + H": "Toggle help panel",
        "Ctrl + D": "Dark/Light mode toggle",
        "Ctrl + S": "Save current visualization",
        "Ctrl + F": "Open search",
        "Ctrl + E": "Export data",
        "Ctrl + P": "Print view",
        "Ctrl + R": "Refresh data"
    }
    
    col1, col2 = st.columns(2)
    
    for i, (key, description) in enumerate(shortcuts.items()):
        if i < len(shortcuts) // 2 + len(shortcuts) % 2:
            with col1:
                st.markdown(f"**{key}**: {description}")
        else:
            with col2:
                st.markdown(f"**{key}**: {description}")
    
    # Version information
    st.markdown("---")
    st.caption("Numera Analytics Platform v1.0.0 | © 2025 Numera, Inc.")


# This function would be called from the main app file
def show_help_page():
    load_help_page()

