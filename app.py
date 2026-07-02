import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Safely load translator
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_READY = True
except ImportError:
    TRANSLATOR_READY = False

st.set_page_config(page_title="TrueScore: Google Maps Filter", layout="wide", page_icon="🍔")

# --- CUSTOM CSS DESIGN ---
st.markdown("""
<style>
    /* Foodie Card Styling */
    .foodie-card {
        background: rgba(30, 30, 40, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .foodie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Rank Badges */
    .rank-badge {
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    .gold-card {
        border-left: 5px solid #f1c40f;
        box-shadow: 0 4px 15px rgba(241, 196, 15, 0.08);
    }
    .gold-card .rank-badge {
        background: rgba(241, 196, 15, 0.15);
        color: #f1c40f;
    }
    
    .silver-card {
        border-left: 5px solid #bdf3ff;
        box-shadow: 0 4px 15px rgba(189, 243, 255, 0.08);
    }
    .silver-card .rank-badge {
        background: rgba(189, 243, 255, 0.15);
        color: #bdf3ff;
    }
    
    .bronze-card {
        border-left: 5px solid #e67e22;
        box-shadow: 0 4px 15px rgba(230, 126, 34, 0.08);
    }
    .bronze-card .rank-badge {
        background: rgba(230, 126, 34, 0.15);
        color: #e67e22;
    }
    
    .overrated-card {
        border-left: 5px solid #e74c3c;
        background: rgba(231, 76, 60, 0.04);
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.08);
    }
    .overrated-card .rank-badge {
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
    }
    
    /* Metrics */
    .restaurant-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .score-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .score-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
    }
    .score-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
    }
    .diff-badge {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .diff-badge.positive {
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
    }
    .diff-badge.negative {
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
    }
    
    .card-footer {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Algorithm Weights")

# Predefined weight configurations
PRESETS = {
    "Default Balanced": {
        "level_0": 1.0, "level_1_3": 1.2, "level_4_6": 2.0, "level_7_plus": 5.0,
        "no_text": 0.2, "short_text": 1.0, "long_text": 1.5, "photo_bonus": 1.0,
        "recent": 1.2, "mid": 1.0, "old": 0.5,
        "bot_penalty": 1.5, "one_star": 1.0, "sentiment_mismatch": 0.2
    },
    "Strict Anti-Fraud": {
        "level_0": 0.5, "level_1_3": 1.0, "level_4_6": 2.0, "level_7_plus": 6.0,
        "no_text": 0.0, "short_text": 0.5, "long_text": 2.0, "photo_bonus": 1.5,
        "recent": 1.5, "mid": 0.8, "old": 0.2,
        "bot_penalty": 3.0, "one_star": 2.0, "sentiment_mismatch": 0.5
    },
    "Foodie Favorite": {
        "level_0": 0.5, "level_1_3": 1.0, "level_4_6": 3.0, "level_7_plus": 8.0,
        "no_text": 0.1, "short_text": 0.8, "long_text": 2.5, "photo_bonus": 2.0,
        "recent": 1.0, "mid": 1.0, "old": 0.8,
        "bot_penalty": 1.0, "one_star": 1.0, "sentiment_mismatch": 0.1
    },
    "Recent Bias": {
        "level_0": 1.0, "level_1_3": 1.2, "level_4_6": 2.0, "level_7_plus": 5.0,
        "no_text": 0.2, "short_text": 1.0, "long_text": 1.5, "photo_bonus": 1.0,
        "recent": 2.0, "mid": 0.5, "old": 0.0,
        "bot_penalty": 1.5, "one_star": 1.0, "sentiment_mismatch": 0.2
    }
}

# Initialize session state keys for all slider parameters
for key, val in PRESETS["Default Balanced"].items():
    if key not in st.session_state:
        st.session_state[key] = val

if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = "Default Balanced"

# Callbacks for two-way synchronization
def on_preset_change():
    preset_name = st.session_state["preset_select"]
    if preset_name != "Custom":
        for key, val in PRESETS[preset_name].items():
            st.session_state[key] = val
        st.session_state["selected_preset"] = preset_name

def on_slider_change():
    current_vals = {key: st.session_state[key] for key in PRESETS["Default Balanced"].keys()}
    matched = "Custom"
    for preset_name, preset_vals in PRESETS.items():
        if all(abs(current_vals[k] - preset_vals[k]) < 0.01 for k in preset_vals):
            matched = preset_name
            break
    st.session_state["selected_preset"] = matched

preset_options = ["Custom", "Default Balanced", "Strict Anti-Fraud", "Foodie Favorite", "Recent Bias"]
curr_preset = st.session_state["selected_preset"]
preset_idx = preset_options.index(curr_preset) if curr_preset in preset_options else 0

preset_select = st.sidebar.selectbox(
    "📋 Algorithm Weight Preset",
    options=preset_options,
    index=preset_idx,
    key="preset_select",
    on_change=on_preset_change
)

st.sidebar.subheader("1. Authority (Local Guide Level)")
w_level_0 = st.sidebar.slider("Level 0 (No Level)", 0.0, 2.0, key="level_0", on_change=on_slider_change)
w_level_1_3 = st.sidebar.slider("Level 1-3", 0.0, 3.0, key="level_1_3", on_change=on_slider_change)
w_level_4_6 = st.sidebar.slider("Level 4-6", 0.0, 5.0, key="level_4_6", on_change=on_slider_change)
w_level_7_plus = st.sidebar.slider("Level 7-10", 1.0, 10.0, key="level_7_plus", on_change=on_slider_change)

st.sidebar.subheader("2. Effort & Proof")
w_no_text = st.sidebar.slider("No Text (Stars Only)", 0.0, 1.0, key="no_text", on_change=on_slider_change)
w_short_text = st.sidebar.slider("Short Text (<100 chars)", 0.0, 2.0, key="short_text", on_change=on_slider_change)
w_long_text = st.sidebar.slider("Detailed Text (>100 chars)", 1.0, 5.0, key="long_text", on_change=on_slider_change)
w_photo_bonus = st.sidebar.slider("Photo Included Bonus (+)", 0.0, 3.0, key="photo_bonus", on_change=on_slider_change)

st.sidebar.subheader("3. Recency")
w_recent = st.sidebar.slider("< 1 Year Old", 0.5, 2.0, key="recent", on_change=on_slider_change)
w_mid = st.sidebar.slider("1-3 Years Old", 0.5, 2.0, key="mid", on_change=on_slider_change)
w_old = st.sidebar.slider("> 3 Years Old", 0.0, 1.0, key="old", on_change=on_slider_change)

st.sidebar.subheader("🚨 4. Anti-Fraud Penalties")
w_bot_penalty = st.sidebar.slider("Ghost Account Penalty", 0.0, 3.0, key="bot_penalty", on_change=on_slider_change)
w_one_star = st.sidebar.slider("Tourist Trap Penalty (1-Star %)", 0.0, 3.0, key="one_star", on_change=on_slider_change)
w_sentiment_mismatch = st.sidebar.slider(
    "Sentiment Mismatch Multiplier", 0.0, 1.0, key="sentiment_mismatch", on_change=on_slider_change,
    help="Multiplier applied to the weight of reviews where the text sentiment contradicts the rating (e.g. 5 stars + negative text). 0.0 discards them, 1.0 ignores the penalty."
)

DEFAULT_BOT_PHRASES = "good, great, nice, amazing, excellent, perfect, love it, ok, best, cool, highly recommend, highly recommended, good food, nice place, great food, great service, friendly staff, five stars, 5 stars, super, awesome, wow"

with st.sidebar.expander("📝 Custom Bot Phrases"):
    bot_phrases_input = st.text_area(
        "List of generic phrases (comma-separated):",
        value=DEFAULT_BOT_PHRASES,
        height=150,
        help="Reviews matching these exactly (case-insensitive, ignoring punctuation) will be flagged as bots if submitted by low guide level reviewers (<= 1)."
    )

import string
def clean_phrase(p):
    p_clean = p.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    return " ".join(p_clean.split())

bot_phrases_list = [clean_phrase(p) for p in bot_phrases_input.split(",") if p.strip()]

st.sidebar.subheader("5. Advanced")
apply_smoothing = st.sidebar.checkbox("Apply Bayesian Smoothing", value=True)

# --- 2. DATA PROCESSING FUNCTION ---
@st.cache_data
def calculate_true_scores(df, weights, apply_smoothing, bot_phrases_list):
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
        
    # Identify and clean category/type column for Cuisines (excluding URL/link columns)
    cat_cols = [
        c for c in df.columns 
        if any(k in c for k in ['cat', 'type', 'cuisine', 'genre', 'tag', 'label', 'class', 'kind'])
        and not any(neg in c for neg in ['link', 'url', 'uri', 'website', 'http', 'href'])
    ]
    if cat_cols:
        cat_series = df[cat_cols[0]].fillna("Unknown").astype(str).str.title()
        for suffix in [" Restaurant", " Cafe", " Shop", " Bar", " Eatery", " Place", " Grill", " Bistro", " Buffet", " Diner", " Joint", " Pub"]:
            cat_series = cat_series.str.replace(suffix, "", case=False, regex=False)
        df['cuisine_category'] = cat_series.str.strip().replace("", "Unknown")
    else:
        df['cuisine_category'] = "Unknown"
        
    if 'months_old' not in df.columns:
        date_cols = [c for c in df.columns if 'date' in c or 'time' in c]
        if date_cols:
            df['date_parsed'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            if df['date_parsed'].dt.tz is not None: df['date_parsed'] = df['date_parsed'].dt.tz_localize(None)
            df['months_old'] = ((pd.Timestamp.now().tz_localize(None) - df['date_parsed']).dt.days / 30)
        else: df['months_old'] = 12

    df['months_old'] = pd.to_numeric(df['months_old'], errors='coerce').fillna(12)
    
    # 🚨 FLAG SUSPICIOUS REVIEWS
    # Heuristic 1: Low Account Activity (Low reviews count)
    df['is_bot_count'] = ((df['rating'] == 5) & (df['author_reviews_count'] <= 3)).astype(int)
    
    # Clean text helper
    import string
    def clean_text_func(txt):
        if not isinstance(txt, str):
            return ""
        txt_clean = txt.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        return " ".join(txt_clean.split())
        
    df['cleaned_review_text'] = df['review_text'].apply(clean_text_func)
    
    # Heuristic 2: Low guide level + generic phrase match
    bot_phrases_set = set(bot_phrases_list)
    df['is_bot_phrase'] = ((df['rating'] == 5) & (df['local_guide_level'] <= 1) & (df['cleaned_review_text'].isin(bot_phrases_set))).astype(int)
    
    df['is_bot'] = (df['is_bot_count'] | df['is_bot_phrase']).astype(int)
    
    # 🚨 SENTIMENT MISMATCH DETECTION (VADER)
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_compound(txt):
        if not isinstance(txt, str) or not txt.strip():
            return 0.0
        try:
            return analyzer.polarity_scores(txt)['compound']
        except Exception:
            return 0.0
            
    df['sentiment_compound'] = df['review_text'].apply(get_sentiment_compound)
    
    df['is_contradictory_positive'] = ((df['rating'] >= 4.0) & (df['sentiment_compound'] <= -0.1)).astype(int)
    df['is_contradictory_negative'] = ((df['rating'] <= 2.0) & (df['sentiment_compound'] >= 0.1)).astype(int)
    df['is_sentiment_mismatch'] = (df['is_contradictory_positive'] | df['is_contradictory_negative']).astype(int)
    
    auth_cond = [(df['local_guide_level'] >= 7), (df['local_guide_level'] >= 4), (df['local_guide_level'] >= 1)]
    df['auth_weight'] = np.select(auth_cond, [weights['l7'], weights['l4'], weights['l1']], default=weights['l0'])
    
    df['text_len'] = df['review_text'].astype(str).str.len()
    effort_cond = [(df['text_len'] == 0), (df['text_len'] < 100), (df['text_len'] >= 100)]
    df['effort_weight'] = np.select(effort_cond, [weights['no_text'], weights['short'], weights['long']], default=1.0)
    df['effort_weight'] = np.where(df['has_photo'], df['effort_weight'] + weights['photo'], df['effort_weight'])
    
    recency_cond = [(df['months_old'] <= 12), (df['months_old'] <= 36)]
    df['recency_weight'] = np.select(recency_cond, [weights['recent'], weights['mid']], default=weights['old'])
                               
    df['final_weight'] = df['auth_weight'] * df['effort_weight'] * df['recency_weight']
    
    # Apply sentiment mismatch discount
    mismatch_multiplier = weights.get('sentiment_mismatch', 0.2)
    df['final_weight'] = np.where(df['is_sentiment_mismatch'] == 1, df['final_weight'] * mismatch_multiplier, df['final_weight'])
    
    df['weighted_rating'] = df['rating'] * df['final_weight']
    
    restaurants = df.groupby(name_col).agg(
        total_weight=('final_weight', 'sum'),
        sum_weighted_rating=('weighted_rating', 'sum'),
        total_reviews=('rating', 'count'),
        google_rating=('rating', 'mean'),
        bot_count=('is_bot', 'sum'),
        one_star_count=('rating', lambda x: (x == 1).sum()),
        cuisine=('cuisine_category', 'first')
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
    
    final = restaurants[[name_col, 'google_rating', 'true_score', 'total_reviews', 'bot_ratio', 'one_star_ratio', 'cuisine']].copy()
    final.rename(columns={
        name_col: 'Restaurant Name', 
        'google_rating': 'Google Avg', 
        'true_score': 'True Score', 
        'total_reviews': 'Reviews',
        'bot_ratio': 'Bot %',
        'one_star_ratio': '1-Star %',
        'cuisine': 'Cuisine'
    }, inplace=True)
    
    final['Score Diff'] = final['True Score'] - final['Google Avg']
    final['Bot %'] = final['Bot %'] * 100
    final['1-Star %'] = final['1-Star %'] * 100
    
    final = final[['Restaurant Name', 'Google Avg', 'True Score', 'Score Diff', 'Bot %', '1-Star %', 'Reviews', 'Cuisine']]
    
    return final.sort_values('True Score', ascending=False), df, None

# --- 3. MAIN UI ---
st.title("🍔 TrueScore: Google Maps Review Filter")
st.markdown("Find the *true* best restaurants, search for specific dishes, and translate local reviews on the fly.")

def generate_demo_data():
    trap_names = ['The Paid Review Trap (Fake 5s, Angry 1s)'] * 222
    gem_names = ['Hidden Gem Kitchen (Real Foodies)'] * 50
    
    # 220 regular trap + 2 contradictory trap reviews
    trap_ratings = [5.0]*160 + [1.0]*40 + [5.0]*20 + [5.0, 1.0]
    trap_guides = np.random.choice([0, 1], 200).tolist() + [0]*20 + [4, 5]
    
    trap_texts = (
        ['']*150 + 
        ['Terrible service, raw food! Avoid at all costs.']*50 + 
        ['Nice!']*5 + ['Great food']*5 + ['Highly recommend!']*5 + ['Love it']*5 +
        [
            "Worst restaurant ever! The food was cold, service was rude, and we got food poisoning. Never coming back!",
            "Absolutely fantastic experience! The duck confit was cooked to perfection, the service was fast, and the atmosphere was lovely. Highly recommend!"
        ]
    )
    trap_photos = [False]*190 + [True]*10 + [False]*20 + [False, True]
    
    trap_author_counts = np.random.choice([1, 2, 3], 200).tolist() + [5]*20 + [12, 20]
    trap_months = np.random.randint(1, 48, 220).tolist() + [6, 8]
    
    gem_ratings = [5.0]*40 + [4.0]*10
    gem_guides = np.random.choice([5, 6, 7, 8], 50).tolist()
    gem_texts = ['Incroyable! Le canard confit est le meilleur de Paris. Le vin naturel était parfait.']*50
    gem_photos = [True]*50
    gem_author_counts = np.random.choice([25, 80, 150], 50).tolist()
    gem_months = np.random.randint(1, 48, 50).tolist()
    
    trap_types = ['Tourist Trap diner'] * 222
    gem_types = ['French bistro'] * 50
    
    return pd.DataFrame({
        'restaurant_name': trap_names + gem_names,
        'rating': trap_ratings + gem_ratings,
        'local_guide_level': trap_guides + gem_guides,
        'review_text': trap_texts + gem_texts,
        'has_photo': trap_photos + gem_photos,
        'author_reviews_count': trap_author_counts + gem_author_counts,
        'months_old': trap_months + gem_months,
        'type': trap_types + gem_types
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
        'bot_penalty': w_bot_penalty, 'one_star': w_one_star,
        'sentiment_mismatch': w_sentiment_mismatch
    }

    results, raw_df, error = calculate_true_scores(df, current_weights, apply_smoothing, bot_phrases_list)
    
    if error:
        st.error(error)
    else:
        # --- SIDEBAR CUISINE FILTER ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Cuisine Filter")
        if 'Cuisine' in results.columns:
            unique_cuisines = sorted(results['Cuisine'].unique().tolist())
            selected_cuisines = st.sidebar.multiselect(
                "Cuisines to show:",
                options=unique_cuisines,
                default=None,
                help="Select one or more cuisines to filter. Leave empty to show all."
            )
            
            # If all cuisines are "Unknown", display an info tip about columns checked
            if len(unique_cuisines) == 1 and unique_cuisines[0] == "Unknown":
                st.sidebar.info(
                    "💡 **Tip:** Cuisine categories show up as 'Unknown' because we couldn't detect a category column in your CSV. "
                    "We check for columns containing `cat`, `type`, `cuisine`, `tag`, etc. "
                    f"Your CSV headers: `{', '.join(df.columns[:8])}`"
                )
            
            # Apply the filter if selections are made
            if selected_cuisines:
                results = results[results['Cuisine'].isin(selected_cuisines)]
                raw_df = raw_df[raw_df['restaurant_name'].isin(results['Restaurant Name'])]
        # --- ANTI-FRAUD STATUS BREAKDOWN ---
        with st.expander("🔍 Anti-Fraud Flag Breakdown", expanded=False):
            total_revs = len(raw_df)
            flagged_count = int(raw_df['is_bot'].sum())
            flagged_count_activity = int(raw_df['is_bot_count'].sum())
            flagged_count_phrase = int(raw_df['is_bot_phrase'].sum())
            flagged_count_sentiment = int(raw_df['is_sentiment_mismatch'].sum()) if 'is_sentiment_mismatch' in raw_df.columns else 0
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Reviews", f"{total_revs:,}")
            c2.metric("Flagged Bots (Total)", f"{flagged_count:,}", f"{(flagged_count/total_revs*100):.1f}% of total" if total_revs > 0 else "0.0%")
            c3.metric("Flagged: Low Activity", f"{flagged_count_activity:,}", f"{(flagged_count_activity/total_revs*100):.1f}%" if total_revs > 0 else "0.0%")
            c4.metric("Flagged: Generic Phrases", f"{flagged_count_phrase:,}", f"{(flagged_count_phrase/total_revs*100):.1f}%" if total_revs > 0 else "0.0%")
            c5.metric("Sentiment Mismatch", f"{flagged_count_sentiment:,}", f"{(flagged_count_sentiment/total_revs*100):.1f}%" if total_revs > 0 else "0.0%")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                # Show a list of flagged reviews for transparency
                phrase_flagged_examples = raw_df[raw_df['is_bot_phrase'] == 1][['restaurant_name', 'rating', 'review_text', 'local_guide_level']].copy()
                if len(phrase_flagged_examples) > 0:
                    st.markdown("### 📝 Sample Reviews Flagged for Generic Phrases")
                    st.dataframe(phrase_flagged_examples.rename(columns={
                        'restaurant_name': 'Restaurant',
                        'rating': 'Stars',
                        'review_text': 'Review Text',
                        'local_guide_level': 'Guide Level'
                    }).head(10), use_container_width=True, hide_index=True)
            with col_b2:
                # Show a list of sentiment mismatch reviews for transparency
                if 'is_sentiment_mismatch' in raw_df.columns:
                    sentiment_flagged_examples = raw_df[raw_df['is_sentiment_mismatch'] == 1][['restaurant_name', 'rating', 'review_text', 'sentiment_compound']].copy()
                    if len(sentiment_flagged_examples) > 0:
                        st.markdown("### 🎭 Sample Reviews Flagged for Sentiment Mismatch")
                        st.dataframe(sentiment_flagged_examples.rename(columns={
                            'restaurant_name': 'Restaurant',
                            'rating': 'Stars',
                            'review_text': 'Review Text',
                            'sentiment_compound': 'Sentiment Score'
                        }).head(10), use_container_width=True, hide_index=True)

        # We divide the UI into Tabs for a cleaner experience
        tab1, tab_insights, tab2, tab3 = st.tabs(["🏆 TrueScore Leaderboard", "📊 Rating Insights", "🥘 Dish Finder", "📖 Read & Translate Reviews"])
        
        # --- TAB 1: LEADERBOARD ---
        with tab1:
            if len(results) > 0:
                card_cols = st.columns(4)
                
                # Rank 1: Gold
                if len(results) >= 1:
                    r1 = results.iloc[0]
                    diff_class = "positive" if r1['Score Diff'] >= 0 else "negative"
                    diff_sign = "+" if r1['Score Diff'] >= 0 else ""
                    with card_cols[0]:
                        st.markdown(f"""
                        <div class="foodie-card gold-card">
                            <div class="rank-badge">🥇 Rank 1</div>
                            <h4 class="restaurant-title" title="{r1['Restaurant Name']}">{r1['Restaurant Name']}</h4>
                            <div class="score-row">
                                <span class="score-label">True Score</span>
                                <span class="score-value">{r1['True Score']:.2f}</span>
                            </div>
                            <div class="score-row" style="margin-bottom: 10px;">
                                <span class="score-label">Google Avg</span>
                                <span class="score-value" style="font-size:1.05rem; font-weight:600; color:rgba(255,255,255,0.7);">{r1['Google Avg']:.2f}</span>
                            </div>
                            <span class="diff-badge {diff_class}">{diff_sign}{r1['Score Diff']:.2f} delta</span>
                            <div class="card-footer">{int(r1['Reviews'])} reviews • {r1['Bot %']:.1f}% bot</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # Rank 2: Silver
                if len(results) >= 2:
                    r2 = results.iloc[1]
                    diff_class = "positive" if r2['Score Diff'] >= 0 else "negative"
                    diff_sign = "+" if r2['Score Diff'] >= 0 else ""
                    with card_cols[1]:
                        st.markdown(f"""
                        <div class="foodie-card silver-card">
                            <div class="rank-badge">🥈 Rank 2</div>
                            <h4 class="restaurant-title" title="{r2['Restaurant Name']}">{r2['Restaurant Name']}</h4>
                            <div class="score-row">
                                <span class="score-label">True Score</span>
                                <span class="score-value">{r2['True Score']:.2f}</span>
                            </div>
                            <div class="score-row" style="margin-bottom: 10px;">
                                <span class="score-label">Google Avg</span>
                                <span class="score-value" style="font-size:1.05rem; font-weight:600; color:rgba(255,255,255,0.7);">{r2['Google Avg']:.2f}</span>
                            </div>
                            <span class="diff-badge {diff_class}">{diff_sign}{r2['Score Diff']:.2f} delta</span>
                            <div class="card-footer">{int(r2['Reviews'])} reviews • {r2['Bot %']:.1f}% bot</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # Rank 3: Bronze
                if len(results) >= 3:
                    r3 = results.iloc[2]
                    diff_class = "positive" if r3['Score Diff'] >= 0 else "negative"
                    diff_sign = "+" if r3['Score Diff'] >= 0 else ""
                    with card_cols[2]:
                        st.markdown(f"""
                        <div class="foodie-card bronze-card">
                            <div class="rank-badge">🥉 Rank 3</div>
                            <h4 class="restaurant-title" title="{r3['Restaurant Name']}">{r3['Restaurant Name']}</h4>
                            <div class="score-row">
                                <span class="score-label">True Score</span>
                                <span class="score-value">{r3['True Score']:.2f}</span>
                            </div>
                            <div class="score-row" style="margin-bottom: 10px;">
                                <span class="score-label">Google Avg</span>
                                <span class="score-value" style="font-size:1.05rem; font-weight:600; color:rgba(255,255,255,0.7);">{r3['Google Avg']:.2f}</span>
                            </div>
                            <span class="diff-badge {diff_class}">{diff_sign}{r3['Score Diff']:.2f} delta</span>
                            <div class="card-footer">{int(r3['Reviews'])} reviews • {r3['Bot %']:.1f}% bot</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with card_cols[2]:
                        st.write("")
                        
                # Most Overrated
                if len(results) >= 2:
                    worst = results.sort_values('Score Diff').iloc[0]
                    if worst['Score Diff'] < 0:
                        with card_cols[3]:
                            st.markdown(f"""
                            <div class="foodie-card overrated-card">
                                <div class="rank-badge">📉 Tourist Trap</div>
                                <h4 class="restaurant-title" title="{worst['Restaurant Name']}">{worst['Restaurant Name']}</h4>
                                <div class="score-row">
                                    <span class="score-label">True Score</span>
                                    <span class="score-value" style="color:#e74c3c;">{worst['True Score']:.2f}</span>
                                </div>
                                <div class="score-row" style="margin-bottom: 10px;">
                                    <span class="score-label">Google Avg</span>
                                    <span class="score-value" style="font-size:1.05rem; font-weight:600; color:rgba(255,255,255,0.7);">{worst['Google Avg']:.2f}</span>
                                </div>
                                <span class="diff-badge negative">{worst['Score Diff']:.2f} delta</span>
                                <div class="card-footer">{int(worst['Reviews'])} reviews • {worst['Bot %']:.1f}% bot</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with card_cols[3]:
                            st.write("")
                else:
                    with card_cols[3]:
                        st.write("")
                        
        # --- TAB: RATING INSIGHTS ---
        with tab_insights:
            if len(results) > 0:
                st.subheader("📊 Interactive Rating Analysis")
                st.markdown("This scatter plot compares the original **Google Average** with our recalculated **True Score**. Points above the diagonal dashed line represent **Hidden Gems** (places that are better than their Google average suggests), and points below represent **Tourist Traps / Overrated** places.")
                
                # Prepare data
                plot_df = results.copy()
                plot_df['Review Count'] = plot_df['Reviews']
                
                fig = px.scatter(
                    plot_df,
                    x="Google Avg",
                    y="True Score",
                    color="Score Diff",
                    size="Review Count",
                    hover_name="Restaurant Name",
                    hover_data={
                        "Google Avg": ":.2f",
                        "True Score": ":.2f",
                        "Score Diff": ":+.2f",
                        "Bot %": ":.1f",
                        "Review Count": True
                    },
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    color_continuous_midpoint=0.0,
                    labels={"Score Diff": "Score Delta"}
                )
                
                # Draw diagonal line
                min_val = min(plot_df['Google Avg'].min(), plot_df['True Score'].min()) - 0.2
                max_val = max(plot_df['Google Avg'].max(), plot_df['True Score'].max()) + 0.2
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                    name="Google Avg = True Score",
                    showlegend=False
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Google Average Rating", gridcolor="rgba(255,255,255,0.08)", range=[min_val, max_val]),
                    yaxis=dict(title="TrueScore rating", gridcolor="rgba(255,255,255,0.08)", range=[min_val, max_val]),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
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
