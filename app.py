import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueScore: Google Maps Filter", layout="wide", page_icon="🍔")

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Algorithm Weights")
st.sidebar.markdown("Tweak how much influence different review traits have on the final score.")

st.sidebar.subheader("1. Authority (Local Guide Level)")
w_level_0 = st.sidebar.slider("Level 0 (No Level)", 0.0, 2.0, 1.0, 0.1)
w_level_1_3 = st.sidebar.slider("Level 1-3", 0.0, 3.0, 1.2, 0.1)
w_level_4_6 = st.sidebar.slider("Level 4-6", 0.0, 5.0, 2.0, 0.1)
w_level_7_plus = st.sidebar.slider("Level 7-10", 1.0, 10.0, 5.0, 0.5)

st.sidebar.subheader("2. Effort & Proof")
w_no_text = st.sidebar.slider("No Text (Stars Only)", 0.0, 1.0, 0.2, 0.1)
w_short_text = st.sidebar.slider("Short Text (<100 chars)", 0.0, 2.0, 1.0, 0.1)
w_long_text = st.sidebar.slider("Detailed Text (>100 chars)", 1.0, 5.0, 1.5, 0.1)
w_photo_bonus = st.sidebar.slider("Photo Included Bonus (+)", 0.0, 3.0, 1.0, 0.1)

st.sidebar.subheader("3. Recency")
w_recent = st.sidebar.slider("< 1 Year Old", 0.5, 2.0, 1.2, 0.1)
w_mid = st.sidebar.slider("1-3 Years Old", 0.5, 2.0, 1.0, 0.1)
w_old = st.sidebar.slider("> 3 Years Old", 0.0, 1.0, 0.5, 0.1)

st.sidebar.subheader("4. Advanced")
apply_smoothing = st.sidebar.checkbox("Apply Bayesian Smoothing", value=True, help="Pulls restaurants with very few reviews toward the average so they don't unfairly dominate.")

# --- 2. DATA PROCESSING FUNCTION ---
@st.cache_data
def calculate_true_scores(df, weights, apply_smoothing):
    df = df.copy()
    
    # Standardize column names (lowercase, replace spaces)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # --- Outscraper Compatibility Formatting ---
    if 'name' in df.columns and 'restaurant_name' not in df.columns: 
        df.rename(columns={'name': 'restaurant_name'}, inplace=True)
    if 'review_rating' in df.columns: 
        df.rename(columns={'review_rating': 'rating'}, inplace=True)
    if 'review_datetime_utc' in df.columns: 
        df.rename(columns={'review_datetime_utc': 'date'}, inplace=True)
        
    # Extract Local Guide Level from Outscraper's author_title column (e.g., "Local Guide · Level 6")
    if 'author_title' in df.columns and 'local_guide_level' not in df.columns:
        df['local_guide_level'] = df['author_title'].astype(str).str.extract(r'Level (\d+)').fillna(0)
        
    # Convert Outscraper image URLs into True/False for the photo bonus
    if 'review_img_url' in df.columns and 'has_photo' not in df.columns:
        df['has_photo'] = df['review_img_url'].notna() & (df['review_img_url'] != '')
    # Identify the restaurant name column safely
    name_col = 'restaurant_name' if 'restaurant_name' in df.columns else df.columns[0]
    
    # Identify rating column safely
    if 'rating' not in df.columns:
        rating_cols = [c for c in df.columns if 'rating' in c or 'score' in c or 'stars' in c]
        if rating_cols: df['rating'] = df[rating_cols[0]]
        else: return None, "Could not find a 'rating' column in your CSV."

    # Graceful handling of missing expected columns
    if 'local_guide_level' not in df.columns: df['local_guide_level'] = 0
    if 'review_text' not in df.columns: df['review_text'] = ''
    
    # Handle Photo Boolean mapping
    if 'has_photo' not in df.columns: 
        df['has_photo'] = False
    elif df['has_photo'].dtype == object:
        df['has_photo'] = df['has_photo'].astype(str).str.lower().isin(['true', 'yes', '1', 't'])
        
    # Handle Dates
    if 'months_old' not in df.columns:
        date_cols = [c for c in df.columns if 'date' in c or 'time' in c]
        if date_cols:
            df['date_parsed'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df['months_old'] = ((pd.Timestamp.now() - df['date_parsed']).dt.days / 30)
        else:
            df['months_old'] = 12

    # Clean NAs
    df['local_guide_level'] = pd.to_numeric(df['local_guide_level'], errors='coerce').fillna(0)
    df['review_text'] = df['review_text'].fillna('')
    df['months_old'] = pd.to_numeric(df['months_old'], errors='coerce').fillna(12)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3)
    
    # 1. Authority Weight
    auth_cond = [
        (df['local_guide_level'] >= 7),
        (df['local_guide_level'] >= 4),
        (df['local_guide_level'] >= 1)
    ]
    df['auth_weight'] = np.select(auth_cond, [weights['l7'], weights['l4'], weights['l1']], default=weights['l0'])
    
    # 2. Effort Weight
    df['text_len'] = df['review_text'].astype(str).str.len()
    effort_cond = [
        (df['text_len'] == 0),
        (df['text_len'] < 100),
        (df['text_len'] >= 100)
    ]
    df['effort_weight'] = np.select(effort_cond, [weights['no_text'], weights['short'], weights['long']], default=1.0)
    df['effort_weight'] = np.where(df['has_photo'], df['effort_weight'] + weights['photo'], df['effort_weight'])
    
    # 3. Recency Weight
    recency_cond = [
        (df['months_old'] <= 12),
        (df['months_old'] <= 36)
    ]
    df['recency_weight'] = np.select(recency_cond, [weights['recent'], weights['mid']], default=weights['old'])
                               
    # Calculate Final Weights
    df['final_weight'] = df['auth_weight'] * df['effort_weight'] * df['recency_weight']
    df['weighted_rating'] = df['rating'] * df['final_weight']
    
    # Aggregate by Restaurant
    restaurants = df.groupby(name_col).agg(
        total_weight=('final_weight', 'sum'),
        sum_weighted_rating=('weighted_rating', 'sum'),
        total_reviews=('rating', 'count'),
        google_rating=('rating', 'mean')
    ).reset_index()
    
    restaurants = restaurants[restaurants['total_weight'] > 0] # Avoid div by zero
    restaurants['custom_avg'] = restaurants['sum_weighted_rating'] / restaurants['total_weight']
    
    if apply_smoothing and len(restaurants) > 1:
        C = restaurants['custom_avg'].mean() 
        M = restaurants['total_weight'].quantile(0.25) 
        restaurants['true_score'] = (
            (restaurants['total_weight'] * restaurants['custom_avg']) + (M * C)
        ) / (restaurants['total_weight'] + M)
    else:
        restaurants['true_score'] = restaurants['custom_avg']
    
    # Format and sort
    final = restaurants[[name_col, 'google_rating', 'true_score', 'total_reviews']].copy()
    final.rename(columns={name_col: 'Restaurant Name', 'google_rating': 'Google Rating', 'true_score': 'True Score', 'total_reviews': 'Total Reviews'}, inplace=True)
    final['Score Diff'] = final['True Score'] - final['Google Rating']
    
    # Rounding
    final['Google Rating'] = final['Google Rating'].round(2)
    final['True Score'] = final['True Score'].round(2)
    final['Score Diff'] = final['Score Diff'].round(2)
    
    return final.sort_values('True Score', ascending=False), None

# --- 3. MAIN UI ---
st.title("🍔 TrueScore: Google Maps Review Filter")
st.markdown("Standard Google Maps treats all reviews equally. This tool applies a custom weighted algorithm to scraped Maps data to find the *true* best restaurants by filtering out bots, tourists, and low-effort 5-star ratings.")

# Demo data generation if no file is present
def generate_demo_data():
    return pd.DataFrame({
        'restaurant_name': ['The Tourist Trap (High Rating, Low Effort)'] * 200 + ['Hidden Gem Kitchen (Lower Rating, High Effort)'] * 50,
        'rating': [5.0]*180 + [1.0]*20 + [4.0]*5 + [5.0]*45,
        'local_guide_level': np.random.choice([0, 1, 2], 200).tolist() + np.random.choice([5, 6, 7, 8], 50).tolist(),
        'review_text': ['']*150 + ['Good']*50 + ['Absolutely incredible flavors, the chef really cares about...']*50,
        'has_photo': [False]*190 + [True]*10 + [True]*40,
        'months_old': np.random.randint(1, 48, 250)
    })

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your scraped reviews CSV", type=["csv"])
with col2:
    st.write("<br>", unsafe_allow_html=True)
    load_demo = st.button("🚀 Load Demo Data to Test App")

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} reviews successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
elif load_demo:
    df = generate_demo_data()
    st.info("Showing Demo Data. Feel free to play with the sliders!")

if df is not None:
    # Pack weights into a dictionary to pass to cached function
    current_weights = {
        'l0': w_level_0, 'l1': w_level_1_3, 'l4': w_level_4_6, 'l7': w_level_7_plus,
        'no_text': w_no_text, 'short': w_short_text, 'long': w_long_text, 'photo': w_photo_bonus,
        'recent': w_recent, 'mid': w_mid, 'old': w_old
    }

    # Run the algorithm
    results, error = calculate_true_scores(df, current_weights, apply_smoothing)
    
    if error:
        st.error(error)
    else:
        st.subheader("🏆 Adjusted Restaurant Rankings")
        
        # Display top metrics
        if len(results) >= 2:
            m1, m2 = st.columns(2)
            top_restaurant = results.iloc[0]
            with m1:
                st.metric(label=f"🥇 #1 True Rated", value=top_restaurant['Restaurant Name'], delta=f"{top_restaurant['True Score']} True Score")
            with m2:
                biggest_loser = results.sort_values('Score Diff').iloc[0]
                st.metric(label=f"📉 Most Overrated", value=biggest_loser['Restaurant Name'], delta=f"{biggest_loser['Score Diff']} Drop")
        
        # Style dataframe (Green for jumping up in rank, Red for dropping)
        def color_diff(val):
            if pd.isna(val): return ''
            color = 'rgba(46, 204, 113, 0.2)' if val > 0 else 'rgba(231, 76, 60, 0.2)' if val < 0 else ''
            return f'background-color: {color}'
        
        st.dataframe(
            results.style.map(color_diff, subset=['Score Diff']), 
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        st.download_button(
            label="📥 Download TrueScore Rankings (CSV)",
            data=results.to_csv(index=False).encode('utf-8'),
            file_name='truescore_rankings.csv',
            mime='text/csv',
        )
else:
    st.info("👆 Upload a CSV file or click 'Load Demo Data' to get started.")

st.caption("How to get data: Use a tool like Outscraper or Apify to export Maps reviews for your area into a CSV.")
