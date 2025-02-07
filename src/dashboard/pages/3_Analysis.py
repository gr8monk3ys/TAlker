import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_posts():
    """Load posts from CSV file."""
    try:
        posts_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "posts.csv")
        if os.path.exists(posts_path):
            df = pd.read_csv(posts_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return None
    except Exception as e:
        st.error(f"Error loading posts: {str(e)}")
        return None

def analyze_post_activity(df):
    """Analyze post activity over time."""
    if df is None or df.empty:
        return
    
    st.subheader("📈 Post Activity Over Time")
    col1, col2 = st.columns(2)
    
    # Posts per day
    with col1:
        daily_posts = df.groupby(df['timestamp'].dt.date).size().reset_index()
        daily_posts.columns = ['date', 'count']
        
        fig = px.line(daily_posts, x='date', y='count',
                     title='Posts per Day',
                     labels={'count': 'Number of Posts', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Posts by hour of day
    with col2:
        hourly_posts = df.groupby(df['timestamp'].dt.hour).size().reset_index()
        hourly_posts.columns = ['hour', 'count']
        
        fig = px.bar(hourly_posts, x='hour', y='count',
                    title='Posts by Hour of Day',
                    labels={'count': 'Number of Posts', 'hour': 'Hour'})
        st.plotly_chart(fig, use_container_width=True)

def analyze_user_engagement(df):
    """Analyze user engagement patterns."""
    if df is None or df.empty:
        return
    
    st.subheader("👥 User Engagement")
    col1, col2 = st.columns(2)
    
    # Top contributors
    with col1:
        user_posts = df['username'].value_counts().head(10)
        fig = px.bar(user_posts, title='Top 10 Contributors',
                    labels={'value': 'Number of Posts', 'index': 'Username'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Post status distribution
    with col2:
        status_dist = df['status'].value_counts()
        fig = px.pie(values=status_dist.values, names=status_dist.index,
                    title='Post Status Distribution')
        st.plotly_chart(fig, use_container_width=True)

def analyze_content(df):
    """Analyze post content and topics."""
    if df is None or df.empty:
        return
    
    st.subheader("📝 Content Analysis")
    
    # Combine all post content
    text = ' '.join(df['content'].astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         min_font_size=10).generate(text)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Average post length
    df['content_length'] = df['content'].astype(str).apply(len)
    avg_length = df['content_length'].mean()
    median_length = df['content_length'].median()
    
    col1, col2 = st.columns(2)
    col1.metric("Average Post Length", f"{avg_length:.0f} characters")
    col2.metric("Median Post Length", f"{median_length:.0f} characters")

def analyze_response_patterns(df):
    """Analyze response patterns and timing."""
    if df is None or df.empty:
        return
    
    st.subheader("⏱️ Response Patterns")
    
    # Calculate response times (if available)
    if 'response_time' in df.columns:
        response_times = pd.to_numeric(df['response_time'], errors='coerce')
        avg_response = response_times.mean()
        median_response = response_times.median()
        
        col1, col2 = st.columns(2)
        col1.metric("Average Response Time", f"{avg_response:.1f} hours")
        col2.metric("Median Response Time", f"{median_response:.1f} hours")
        
        # Response time distribution
        fig = px.histogram(df, x='response_time',
                          title='Response Time Distribution',
                          labels={'response_time': 'Response Time (hours)'})
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📊 Course Analytics")
    st.markdown("""
        This page provides detailed analytics and insights about course activity and engagement patterns.
        Use these insights to understand student participation and identify areas that need attention.
    """)
    
    # Load data
    df = load_posts()
    if df is None:
        st.warning("No posts data available. Please check the data directory.")
        return
    
    # Display basic stats
    total_posts = len(df)
    unique_users = df['username'].nunique()
    active_posts = df[df['status'] == 'active'].shape[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", total_posts)
    col2.metric("Unique Users", unique_users)
    col3.metric("Active Posts", active_posts)
    
    # Add tab-based navigation for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Activity Analysis",
        "👥 User Engagement",
        "📝 Content Analysis",
        "⏱️ Response Patterns"
    ])
    
    with tab1:
        analyze_post_activity(df)
    
    with tab2:
        analyze_user_engagement(df)
    
    with tab3:
        analyze_content(df)
    
    with tab4:
        analyze_response_patterns(df)

if __name__ == "__main__":
    main()
