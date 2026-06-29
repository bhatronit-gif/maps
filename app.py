import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weighted Review Calculator", layout="centered")
st.title("⭐️ Google Maps Weighted Review Calculator")
st.write("Filter out the noise from 1-review accounts and see the true rankings.")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload your exported Google Reviews CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns
    required_cols = ['restaurant_name', 'review_rating', 'author_reviews_count']
    if all(col in df.columns for col in required_cols):
        
        # Weighting Logic
        def assign_contributor_weight(count):
            if pd.isna(count) or count <= 2: return 1
            elif count <= 10: return 2
            elif count <= 50: return 3
            else: return 5

        df['review_weight'] = df['author_reviews_count'].apply(assign_contributor_weight)
        df['weighted_rating_product'] = df['review_rating'] * df['review_weight']

        # Calculate Adjusted Score
        adjusted_rankings = df.groupby('restaurant_name').apply(
            lambda x: x['weighted_rating_product'].sum() / x['review_weight'].sum()
        ).reset_index(name='Custom Weighted Score')
        
        final_report = adjusted_rankings.sort_values(by='Custom Weighted Score', ascending=False)
        
        # Display Results
        st.subheader("--- Adjusted Restaurant Rankings ---")
        st.dataframe(final_report, use_container_width=True)
        st.bar_chart(data=final_report, x='restaurant_name', y='Custom Weighted Score')
    else:
        st.error(f"Your CSV must include these exact columns: {required_cols}")