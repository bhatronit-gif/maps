# 🍔 TrueScore: Google Maps Review Filter

Standard Google Maps averages treat low-effort bots and high-effort local foodies as equals. This Streamlit app allows you to upload scraped Google Maps data and recalculate restaurant ratings based on:
- **Reviewer Authority** (Local Guide Level)
- **Review Effort** (Length of text & included photos)
- **Recency** (Penalizing very old reviews)
- **Bayesian Smoothing** (Preventing places with 2 reviews from getting a perfect 5.0)

## Required CSV Format
Export your data using a tool like Outscraper or Apify. Ensure your CSV contains these headers (names can vary slightly, the app will try to auto-detect):
`restaurant_name`, `rating`, `local_guide_level`, `review_text`, `has_photo`, `months_old`
