import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TrueScore: Google Maps Filter", layout="wide", page_icon="🍔")

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Algorithm Weights")

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

st.sidebar.subheader("🚨 4. Anti-Fraud Penalties")
st.sidebar.markdown("Actively subtracts stars from places that buy fake reviews.")
w_bot_penalty = st.sidebar.slider("Ghost Account Penalty", 0.0, 3.0, 1.5, 0.1, help="Max stars to subtract if reviews are overwhelmingly from ghost accounts (< 4 lifetime reviews).")
w_one_star = st.sidebar.slider("Tourist Trap Penalty (1-Star %)", 0.0, 3.0, 1.0, 0.1, help="Max stars to subtract if the restaurant has a high ratio of 1-star reviews.")

st.sidebar.subheader("5. Advanced")
apply_smoothing = st.sidebar.checkbox("Apply Bayesian Smoothing", value=True)

# --- 2. DATA PROCESSING FUNCTION ---
@st.cache_data
def calculate_true_scores(df, weights, apply_smoothing):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # --- Outscraper Compatibility Formatting ---
    if 'review_rating' in df.columns:
        if 'rating' in df.columns:
            df = df.drop(columns=['rating'])
        df.rename(columns={'review_rating': 'rating'}, inplace=True)
        
    if 'name' in df.columns and 'restaurant_name' not in df.columns: 
        df.rename(columns={'name': 'restaurant_name'}, inplace=True)
        
    if 'review_datetime_utc' in df.columns and 'date' not in df.columns: 
        df.rename(columns={'review_datetime_utc': 'date'}, inplace=True)
        
    if 'author_title' in df.columns and 'local_guide_level' not in df.columns:
        df['local_guide_level'] = df['author_title'].astype(str).str.extract(r'Level (\d+)', expand=False).fillna(0)
        
    if 'review_img_url' in df.columns and 'has_photo' not in df.columns:
        df['has_photo'] = df['review_img_url'].notna() & (df['review_img_url'].astype(str).str.strip() != '')
        
    # NEW: Find Author Review Count for Bot Defense
    if 'author_reviews_count' not in df.columns:
        rev_cols = [c for c in df.columns if 'reviews_count' in c or 'author_reviews' in c]
        if rev_cols: df['author_reviews_count'] = df[rev_cols[0]]
        else: df['author_reviews_count'] = 10 
    # -----------------------------------------------
    
    name_col = 'restaurant_name' if 'restaurant_name' in df.columns else df.columns[0]
    
    if 'rating' not in df.columns:
        rating_cols = [c for c in df.columns if 'rating' in c or 'score' in c or 'stars' in c]
        if rating_cols: df['rating'] = df[rating_cols[0]]
        else: return None, "Could not find a 'rating' column in your CSV."
        
    if isinstance(df['rating'], pd.DataFrame):
        df['rating'] = df['rating'].iloc[:, 0]

    df['local_guide_level'] = pd.to_numeric(df['local_guide_level'], errors='coerce').fillna(0)
    if 'review_text' not in df.columns: df['review_text'] = ''
    df['review_text'] = df['review_text'].fillna('')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3)
    df['author_reviews_count'] = pd.to_numeric(df['author_reviews_count'], errors='coerce').fillna(10)
    
    if 'has_photo' not in df.columns: df['has_photo'] = False
    elif df['has_photo'].dtype == object:
        df['has_photo'] = df['has_photo'].astype(str).str.lower().isin(['true', 'yes', '1', 't'])
        
    if 'months_old' not in df.columns:
        date_cols = [c for c in df.columns if 'date' in c or 'time' in c]
        if date_cols:
            df['date_parsed'] = pd.to_datetime(df[date_cols[0]], errors='coerce').dt.tz_localize(None)
            df['months_old'] = ((pd.Timestamp.now().tz_localize(None) - df['date_parsed']).dt.days / 30)
        else:
            df['months_old'] = 12

    df['months_old'] = pd.to_numeric(df['months_old'], errors='coerce').fillna(12)
    
    # 🚨 FLAG SUSPICIOUS REVIEWS (5-stars from ghost accounts <=3 lifetime reviews)
    df['is_bot'] = ((df['rating'] == 5) & (df['author_reviews_count'] <= 3)).astype(int)
    
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
        google_rating=('rating', 'mean'),
        bot_count=('is_bot', 'sum'),
        one_star_count=('rating', lambda x: (x == 1).sum())
    ).reset_index()
    
    restaurants = restaurants[restaurants['total_weight'] > 0]
    
    # Calculate Fraud Metrics
    restaurants['bot_ratio'] = restaurants['bot_count'] / restaurants['total_reviews']
    restaurants['one_star_ratio'] = restaurants['one_star_count'] / restaurants['total_reviews']
    
    # Calculate Custom Average
    restaurants['custom_avg'] = restaurants['sum_weighted_rating'] / restaurants['total_weight']
    
    # Apply Smoothing
    if apply_smoothing and len(restaurants) > 1:
        C = restaurants['custom_avg'].mean() 
        M = restaurants['total_weight'].quantile(0.25) 
        restaurants['true_score'] = (
            (restaurants['total_weight'] * restaurants['custom_avg']) + (M * C)
        ) / (restaurants['total_weight'] + M)
    else:
        restaurants['true_score'] = restaurants['custom_avg']
        
    # --- APPLY FRAUD PENALTIES ---
    restaurants['penalty_applied'] = (restaurants['bot_ratio'] * weights['bot_penalty']) + (restaurants['one_star_ratio'] * weights['one_star'])
    restaurants['true_score'] = restaurants['true_score'] - restaurants['penalty_applied']
    
    # Format and sort
    final = restaurants[[name_col, 'google_rating', 'true_score', 'total_reviews', 'bot_ratio', 'one_star_ratio']].copy()
    final.rename(columns={
        name_col: 'Restaurant Name', 
        'google_rating': 'Google Avg', 
        'true_score': 'True Score', 
        'total_reviews': 'Reviews',
        'bot_ratio': 'Bot %',
        'one_star_ratio': '1-Star %'
    }, inplace=True)
    
    final['Score Diff'] = final['True Score'] - final['Google Avg']
    
   # --- Rounding & Formatting ---
    final['Google Avg'] = final['Google Avg'].round(2)
    final['True Score'] = final['True Score'].round(2)
    final['Score Diff'] = final['Score Diff'].round(2)
    
    # Sort by True Score FIRST (as per your main requirement)
    final = final.sort_values('True Score', ascending=False)
    
    # NOW convert to display-friendly strings
    final['Bot %'] = (final['Bot %'] * 100).round(1).astype(str) + '%'
    final['1-Star %'] = (final['1-Star %'] * 100).round(1).astype(str) + '%'
    
    # Reorder columns
    final = final[['Restaurant Name', 'Google Avg', 'True Score', 'Score Diff', 'Bot %', '1-Star %', 'Reviews']]
    
    return final.sort_values('True Score', ascending=False), None

# --- 3. MAIN UI ---
st.title("🍔 TrueScore: Google Maps Review Filter")
st.markdown("Standard Google Maps treats all reviews equally. This tool applies a custom weighted algorithm to find the *true* best restaurants, and actively punishes spots using fake review farms.")

def generate_demo_data():
    return pd.DataFrame({
        'restaurant_name': ['The Paid Review Trap (Fake 5s, Angry 1s)'] * 200 + ['Hidden Gem Kitchen (Real Foodies)'] * 50,
        'rating': [5.0]*160 + [1.0]*40 + [5.0]*40 + [4.0]*10,
        'local_guide_level': np.random.choice([0, 1], 200).tolist() + np.random.choice([5, 6, 7, 8], 50).tolist(),
        'review_text': ['']*150 + ['Terrible service, raw food!']*50 + ['Absolutely incredible flavors, the chef really cares about...']*50,
        'has_photo': [False]*190 + [True]*10 + [True]*50,
        'author_reviews_count': np.random.choice([1, 2, 3], 200).tolist() + np.random.choice([25, 80, 150], 50).tolist(),
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
    st.info("Showing Demo Data. Look at the Bot % and 1-Star % of the Tourist Trap!")

if df is not None:
    current_weights = {
        'l0': w_level_0, 'l1': w_level_1_3, 'l4': w_level_4_6, 'l7': w_level_7_plus,
        'no_text': w_no_text, 'short': w_short_text, 'long': w_long_text, 'photo': w_photo_bonus,
        'recent': w_recent, 'mid': w_mid, 'old': w_old,
        'bot_penalty': w_bot_penalty, 'one_star': w_one_star
    }

    results, error = calculate_true_scores(df, current_weights, apply_smoothing)
    
    if error:
        st.error(error)
    else:
        st.subheader("🏆 Adjusted Restaurant Rankings")
        
        if len(results) >= 2:
            m1, m2 = st.columns(2)
            top_restaurant = results.iloc[0]
            with m1:
                st.metric(label=f"🥇 #1 True Rated", value=top_restaurant['Restaurant Name'], delta=f"{top_restaurant['True Score']} True Score")
            with m2:
                biggest_loser = results.sort_values('Score Diff').iloc[0]
                st.metric(label=f"📉 Most Overrated", value=biggest_loser['Restaurant Name'], delta=f"{biggest_loser['Score Diff']} Drop")
        
        def color_diff(val):
            if pd.isna(val) or isinstance(val, str): return ''
            color = 'rgba(46, 204, 113, 0.2)' if val > 0 else 'rgba(231, 76, 60, 0.2)' if val < 0 else ''
            return f'background-color: {color}'
        
        st.dataframe(
            results.style.map(color_diff, subset=['Score Diff']), 
            use_container_width=True,
            hide_index=True
        )
