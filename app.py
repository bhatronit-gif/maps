import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse

# Safely load translator
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_READY = True
except ImportError:
    TRANSLATOR_READY = False

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
w_bot_penalty = st.sidebar.slider("Ghost Account Penalty", 0.0, 3.0, 1.5, 0.1)
w_one_star = st.sidebar.slider("Tourist Trap Penalty (1-Star %)", 0.0, 3.0, 1.0, 0.1)

st.sidebar.subheader("5. Advanced")
apply_smoothing = st.sidebar.checkbox("Apply Bayesian Smoothing", value=True)

# --- 2. DATA PROCESSING FUNCTION ---
@st.cache_data
def calculate_true_scores(df, weights, apply_smoothing):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Optional: Outscraper sometimes includes translated text as a column. We can use it if it's there!
    if 'review_translated_text' in df.columns:
        df['review_text'] = df['review_translated_text'].fillna(df.get('review_text', ''))
        
    if 'review_rating' in df.columns:
        if 'rating' in df.columns: df = df.drop(columns=['rating'])
        df.rename(columns={'review_rating': 'rating'}, inplace=True)
        
    if 'name' in df.columns and 'restaurant_name' not in df.columns: 
        df.rename(columns={'name': 'restaurant_name'}, inplace=True)
        
    if 'review_datetime_utc' in df.columns and 'date' not in df.columns: 
        df.rename(columns={'review_datetime_utc': 'date'}, inplace=True)
        
    if 'author_title' in df.columns and 'local_guide_level' not in df.columns:
        df['local_guide_level'] = df['author_title'].astype(str).str.extract(r'(\d+)', expand=False).fillna(0)
        
    if 'review_img_url' in df.columns and 'has_photo' not in df.columns:
        df['has_photo'] = df['review_img_url'].notna() & (df['review_img_url'].astype(str).str.strip() != '')
        
    if 'author_reviews_count' not in df.columns:
        rev_cols = [c for c in df.columns if 'reviews_count' in c or 'author_reviews' in c]
        if rev_cols: df['author_reviews_count'] = df[rev_cols[0]]
        else: df['author_reviews_count'] = 10 
    
    name_col = 'restaurant_name' if 'restaurant_name' in df.columns else df.columns[0]
    if name_col != 'restaurant_name':
        df.rename(columns={name_col: 'restaurant_name'}, inplace=True)
        name_col = 'restaurant_name'
        
    if 'rating' not in df.columns:
        rating_cols = [c for c in df.columns if 'rating' in c or 'score' in c or 'stars' in c]
        if rating_cols: df['rating'] = df[rating_cols[0]]
        else: return None, None, "Could not find a 'rating' column in your CSV."
        
    if isinstance(df['rating'], pd.DataFrame): df['rating'] = df['rating'].iloc[:, 0]

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
            df['date_parsed'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            if df['date_parsed'].dt.tz is not None: df['date_parsed'] = df['date_parsed'].dt.tz_localize(None)
            df['months_old'] = ((pd.Timestamp.now().tz_localize(None) - df['date_parsed']).dt.days / 30)
        else: df['months_old'] = 12

    df['months_old'] = pd.to_numeric(df['months_old'], errors='coerce').fillna(12)
    
    # 🚨 FLAG SUSPICIOUS REVIEWS
    df['is_bot'] = ((df['rating'] == 5) & (df['author_reviews_count'] <= 3)).astype(int)
    
    auth_cond = [(df['local_guide_level'] >= 7), (df['local_guide_level'] >= 4), (df['local_guide_level'] >= 1)]
    df['auth_weight'] = np.select(auth_cond, [weights['l7'], weights['l4'], weights['l1']], default=weights['l0'])
    
    df['text_len'] = df['review_text'].astype(str).str.len()
    effort_cond = [(df['text_len'] == 0), (df['text_len'] < 100), (df['text_len'] >= 100)]
    df['effort_weight'] = np.select(effort_cond, [weights['no_text'], weights['short'], weights['long']], default=1.0)
    df['effort_weight'] = np.where(df['has_photo'], df['effort_weight'] + weights['photo'], df['effort_weight'])
    
    recency_cond = [(df['months_old'] <= 12), (df['months_old'] <= 36)]
    df['recency_weight'] = np.select(recency_cond, [weights['recent'], weights['mid']], default=weights['old'])
                               
    df['final_weight'] = df['auth_weight'] * df['effort_weight'] * df['recency_weight']
    df['weighted_rating'] = df['rating'] * df['final_weight']
    
    restaurants = df.groupby(name_col).agg(
        total_weight=('final_weight', 'sum'),
        sum_weighted_rating=('weighted_rating', 'sum'),
        total_reviews=('rating', 'count'),
        google_rating=('rating', 'mean'),
        bot_count=('is_bot', 'sum'),
        one_star_count=('rating', lambda x: (x == 1).sum())
    ).reset_index()
    
    restaurants = restaurants[restaurants['total_weight'] > 0]
    
    restaurants['bot_ratio'] = restaurants['bot_count'] / restaurants['total_reviews']
    restaurants['one_star_ratio'] = restaurants['one_star_count'] / restaurants['total_reviews']
    restaurants['custom_avg'] = restaurants['sum_weighted_rating'] / restaurants['total_weight']
    
    if apply_smoothing and len(restaurants) > 1:
        C = restaurants['custom_avg'].mean() 
        M = restaurants['total_weight'].quantile(0.25) 
        restaurants['true_score'] = ((restaurants['total_weight'] * restaurants['custom_avg']) + (M * C)) / (restaurants['total_weight'] + M)
    else:
        restaurants['true_score'] = restaurants['custom_avg']
        
    restaurants['penalty_applied'] = (restaurants['bot_ratio'] * weights['bot_penalty']) + (restaurants['one_star_ratio'] * weights['one_star'])
    restaurants['true_score'] = restaurants['true_score'] - restaurants['penalty_applied']
    
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
    final['Bot %'] = final['Bot %'] * 100
    final['1-Star %'] = final['1-Star %'] * 100
    
    final = final[['Restaurant Name', 'Google Avg', 'True Score', 'Score Diff', 'Bot %', '1-Star %', 'Reviews']]
    
    return final.sort_values('True Score', ascending=False), df, None

# --- 3. MAIN UI ---
st.title("🍔 TrueScore: Google Maps Review Filter")
st.markdown("Find the *true* best restaurants, search for specific dishes, and translate local reviews on the fly.")

def generate_demo_data():
    return pd.DataFrame({
        'restaurant_name': ['The Paid Review Trap (Fake 5s, Angry 1s)'] * 200 + ['Hidden Gem Kitchen (Real Foodies)'] * 50,
        'rating': [5.0]*160 + [1.0]*40 + [5.0]*40 + [4.0]*10,
        'local_guide_level': np.random.choice([0, 1], 200).tolist() + np.random.choice([5, 6, 7, 8], 50).tolist(),
        'review_text': ['']*150 + ['Terrible service, raw food! Avoid at all costs.']*50 + ['Incroyable! Le canard confit est le meilleur de Paris. Le vin naturel était parfait.']*50,
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
    st.info("Showing Demo Data. Check out the Dish Finder and Translation features below!")

if df is not None:
    current_weights = {
        'l0': w_level_0, 'l1': w_level_1_3, 'l4': w_level_4_6, 'l7': w_level_7_plus,
        'no_text': w_no_text, 'short': w_short_text, 'long': w_long_text, 'photo': w_photo_bonus,
        'recent': w_recent, 'mid': w_mid, 'old': w_old,
        'bot_penalty': w_bot_penalty, 'one_star': w_one_star
    }

    results, raw_df, error = calculate_true_scores(df, current_weights, apply_smoothing)
    
    if error:
        st.error(error)
    else:
        # We divide the UI into Tabs for a cleaner experience
        tab1, tab2, tab3 = st.tabs(["🏆 TrueScore Leaderboard", "🥘 Dish Finder", "📖 Read & Translate Reviews"])
        
        # --- TAB 1: LEADERBOARD ---
        with tab1:
            if len(results) >= 2:
                m1, m2 = st.columns(2)
                top_restaurant = results.iloc[0]
                with m1:
                    st.metric(label=f"🥇 #1 True Rated", value=top_restaurant['Restaurant Name'], delta=f"{top_restaurant['True Score']:.2f} True Score")
                with m2:
                    biggest_loser = results.sort_values('Score Diff').iloc[0]
                    st.metric(label=f"📉 Most Overrated", value=biggest_loser['Restaurant Name'], delta=f"{biggest_loser['Score Diff']:.2f} Drop")
            
            def color_diff(val):
                if pd.isna(val) or isinstance(val, str): return ''
                color = 'rgba(46, 204, 113, 0.2)' if val > 0 else 'rgba(231, 76, 60, 0.2)' if val < 0 else ''
                return f'background-color: {color}'
            
            st.dataframe(
                results.style.map(color_diff, subset=['Score Diff']), 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Google Avg": st.column_config.NumberColumn(format="%.2f"),
                    "True Score": st.column_config.NumberColumn(format="%.2f"),
                    "Score Diff": st.column_config.NumberColumn(format="%+.2f"), 
                    "Bot %": st.column_config.NumberColumn(format="%.1f%%"),     
                    "1-Star %": st.column_config.NumberColumn(format="%.1f%%"),  
                }
            )
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=results.to_csv(index=False).encode('utf-8'),
                file_name='truescore_rankings.csv',
                mime='text/csv',
            )

        # --- TAB 2: DISH FINDER ---
        with tab2:
            st.subheader("🥘 Find the best specific dish")
            st.markdown("Craving something? Type it below. We will scan all verified reviews and build a mini-leaderboard based solely on who makes *that exact dish* best.")
            st.info("💡 **Pro-Tip:** If searching in a foreign city, use both English and local words separated by a `|` (e.g. `duck|canard`, `snails|escargot`, `wine|vin`)")
            
            dish_query = st.text_input("🔍 What are you looking for?")
            
            if dish_query:
                # Use regex to search, handling multiple terms
                search_pattern = f"(?i)({dish_query})"
                
                # Filter raw reviews containing the dish
                dish_df = raw_df[raw_df['review_text'].astype(str).str.contains(search_pattern, regex=True, na=False)].copy()
                
                if len(dish_df) > 0:
                    st.success(f"Found {len(dish_df)} high-effort reviews mentioning this!")
                    
                    # Score restaurants ONLY based on the reviews that mention this dish
                    dish_rest = dish_df.groupby('restaurant_name').agg(
                        mentions=('rating', 'count'),
                        dish_avg_rating=('rating', 'mean'),
                        total_weight=('final_weight', 'sum'),
                        weighted_score=('weighted_rating', 'sum')
                    ).reset_index()
                    
                    dish_rest['True Dish Score'] = (dish_rest['weighted_score'] / dish_rest['total_weight']).round(2)
                    dish_rest['Avg Rating'] = dish_rest['dish_avg_rating'].round(2)
                    
                    # Merge with overall score
                    dish_leaderboard = pd.merge(dish_rest, results[['Restaurant Name', 'True Score']], left_on='restaurant_name', right_on='Restaurant Name')
                    dish_leaderboard = dish_leaderboard.sort_values('True Dish Score', ascending=False)
                    
                    st.dataframe(
                        dish_leaderboard[['Restaurant Name', 'True Dish Score', 'Avg Rating', 'mentions', 'True Score']].rename(columns={'mentions': 'Total Mentions', 'True Score': 'Overall Restaurant Score'}),
                        use_container_width=True, hide_index=True
                    )
                    
                    st.divider()
                    st.write(f"### Top Trusted Reviews mentioning '{dish_query}'")
                    
                    # Show the best 3 excerpts
                    for _, row in dish_df.sort_values('final_weight', ascending=False).head(3).iterrows():
                        with st.chat_message("user"):
                            st.markdown(f"📍 **{row['restaurant_name']}** ({'⭐'*int(row['rating'])})")
                            
                            review_text_str = str(row["review_text"])
                            
                            # Bold the matching keyword
                            highlighted = re.sub(search_pattern, r'**\1**', review_text_str)
                            st.write(highlighted)
                            
                            encoded_text = urllib.parse.quote(review_text_str)
                            translate_url = f"https://translate.google.com/?sl=auto&tl=en&text={encoded_text}&op=translate"
                            st.markdown(f"[🌐 Translate this review]({translate_url})")
                            st.caption(f"Algorithm Weight: {row['final_weight']:.2f}x")
                else:
                    st.warning("No reviews found mentioning that specific dish.")

        # --- TAB 3: REVIEW TRANSLATOR ---
        with tab3:
            st.subheader("📖 Read Deep-Dive Reviews")
            st.markdown("Select a restaurant below to read its reviews, sorted by **Trust Weight**.")
            
            restaurant_list = results['Restaurant Name'].tolist()
            selected_restaurant = st.selectbox("Choose a restaurant to investigate:", restaurant_list)
            
            if not TRANSLATOR_READY:
                st.info("💡 To enable in-app auto-translation, add `deep-translator>=1.11.4` to your `requirements.txt` file in GitHub.")
            
            translate_toggle = st.toggle("🌐 Auto-Translate reviews to English", value=TRANSLATOR_READY)
            
            if selected_restaurant:
                restaurant_reviews = raw_df[raw_df['restaurant_name'] == selected_restaurant].copy()
                top_reviews = restaurant_reviews.sort_values('final_weight', ascending=False)
                
                count = 0
                for _, row in top_reviews.iterrows():
                    review_text = str(row['review_text']).strip()
                    if review_text == '' or review_text.lower() == 'nan':
                        continue
                    
                    count += 1
                    if count > 5:  # Limit to 5 to save API translation time & screen space
                        break
                        
                    with st.chat_message("user"):
                        stars = '⭐' * int(row['rating'])
                        guide = f" | 🏅 Level {int(row['local_guide_level'])} Guide" if row['local_guide_level'] > 0 else ""
                        months = f" | 📅 {int(row['months_old'])} months ago" if pd.notna(row['months_old']) else ""
                        
                        st.markdown(f"**{stars}{guide}{months}**")
                        
                        display_text = review_text
                        translated = False
                        
                        if translate_toggle and TRANSLATOR_READY:
                            try:
                                # Auto-detect and translate to English
                                display_text = GoogleTranslator(source='auto', target='en').translate(review_text)
                                translated = True
                            except Exception:
                                display_text = review_text + "\n\n*(In-app translation failed).* "
                        
                        st.write(display_text)
                        
                        # Add a quick Google Translate link as fallback or standard
                        if not translated:
                            encoded_text = urllib.parse.quote(review_text)
                            translate_url = f"https://translate.google.com/?sl=auto&tl=en&text={encoded_text}&op=translate"
                            st.markdown(f"[🌐 Open in Google Translate]({translate_url})")
                        
                        details = []
                        if row['has_photo']:
                            details.append("📸 *Included photos*")
                        if row['is_bot'] == 1:
                            details.append("🚨 *Ghost Account*")
                        details.append(f"*Algorithm Weight: {row['final_weight']:.2f}x*")
                        
                        st.caption(" • ".join(details))
                
                if count == 0:
                    st.info("No text reviews available for this restaurant.")
