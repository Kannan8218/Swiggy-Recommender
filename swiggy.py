import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler

class SwiggyRecommender:
    def __init__(self):
        try:
            self.data, self.data_clean, self.encoded_data = None, None, None
            self.city_encoder, self.city_encoded_df = None, None
            self.cuisine_encoder, self.cuisine_encoded_df = None, None
            self.features = None
            st.set_page_config(page_title="Restaurant Recommender", layout="centered")
            st.title("🍽️ Swiggy Restaurant Recommendation System")
        except Exception as e:
            st.error(f"❌ Error in __init__: {e}")
    
    def load_data(self):
        try:
            self.data = pd.read_csv("swiggy.csv", na_values="--")
            self.data = self.data.dropna(subset=['name', 'cuisine'])
            self.data = self.data[self.data['cuisine'] != "8:15 To 11:30 Pm"]
            self.data['city'] = self.data['city'].apply(lambda x: x.lower() if isinstance(x, str) else x)
            self.data['cuisine'] = self.data['cuisine'].apply(lambda x: x.lower() if isinstance(x, str) else x)
            self.data['rating'] = self.data['rating'].fillna(0.5)
            
            rating_map = {
                'Too Few Ratings': 1, '50+ ratings': 51, '100+ ratings': 101, '20+ ratings': 21,
                '500+ ratings': 501, '1K+ ratings': 1001, '5K+ ratings': 5001, '10K+ ratings': 10001
            }
            self.data['rating_count'] = self.data['rating_count'].map(rating_map).fillna(0).astype(int)
            self.data['cost'] = (self.data['cost']
                                 .astype(str)
                                 .str.replace("₹", "", regex=False)
                                 .str.replace(",", "", regex=False)
                                 .astype(float)
                                 .fillna(0.0))
            
            unwanted = [
                "Code valid on bill over Rs.99", "Combo", "Default", "Discount offer from Garden Cafe Express Kankurgachi",
                "Free Delivery ! Limited Stocks!", "MAX 2 Combos per Order!", "Popular Brand Store",
                "Special Discount from (Hotel Swagath)", "Use Code JUMBO30 to avail", "Use code XPRESS121 to avail."
            ]
            self.data['cuisine'] = self.data['cuisine'].str.split(',').apply(
                lambda lst: [c.strip() for c in lst if c.strip() not in unwanted and c.strip() != '']
            )
            
            self.data.to_csv("clean_data.csv", index=False)
            self.data_clean = self.data.copy()
        except Exception as e:
            st.error(f"❌ Error in load_data: {e}")

    def encode_data(self):
        try:
            self.city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.city_encoded = self.city_encoder.fit_transform(self.data_clean[['city']])
            self.city_encoded_df = pd.DataFrame(self.city_encoded, columns=self.city_encoder.get_feature_names_out(['city']))
            
            self.cuisine_encoder = MultiLabelBinarizer()
            self.cuisine_encoded = self.cuisine_encoder.fit_transform(self.data_clean['cuisine'])
            self.cuisine_encoded_df = pd.DataFrame(self.cuisine_encoded, columns=self.cuisine_encoder.classes_)
            
            self.features = pd.concat([
                self.data_clean[['id', 'rating', 'cost', 'rating_count']].reset_index(drop=True),
                self.city_encoded_df.reset_index(drop=True),
                self.cuisine_encoded_df.reset_index(drop=True)
            ], axis=1).fillna(0)
            
            self.features.to_csv("encode_data.csv", index=False)
            with open("encoder.pkl", "wb") as f:
                pickle.dump((self.city_encoder, self.cuisine_encoder), f)
        except Exception as e:
            st.error(f"❌ Error in encode_data: {e}")

    def train_model(self):
        try:
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(self.features.drop(columns=['id']))
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            self.data_clean['cluster'] = kmeans.fit_predict(features_scaled)
        except Exception as e:
            st.error(f"❌ Error in train_model: {e}")

    def recommend(self, city_input, cuisine_list, min_rating, max_cost):
        try:
            filtered = self.data_clean.copy()
            if city_input:
                filtered = filtered[filtered['city'] == city_input.lower()]
            if cuisine_list:
                filtered = filtered[filtered['cuisine'].apply(lambda c: any(cuisine in c for cuisine in cuisine_list))]
            filtered = filtered[(filtered['rating'] >= min_rating) & (filtered['cost'] <= max_cost)]
            
            return filtered[['name', 'cuisine', 'rating', 'cost']].sort_values(by='rating', ascending=False)
        except Exception as e:
            st.error(f"❌ Error in recommend: {e}")
            return pd.DataFrame()

    def user_input(self):
        try:
            # City dropdown
            city_input = st.selectbox("Select your city", options=sorted(self.data_clean['city'].dropna().unique()))
            
            # Cuisine multiselect (use original cuisines, not encoded)
            cuisine_options = sorted(set(c for cuisines in self.data_clean['cuisine'] for c in cuisines))
            cuisine_list = st.multiselect("Choose preferred cuisines", cuisine_options)
            
            # Rating slider
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
            
            # Max cost input
            max_cost = st.number_input("Maximum Cost", min_value=0.0, value=500.0, step=50.0)

            if st.button("Get Recommendations"):
                self.encode_data()
                self.train_model()
                results = self.recommend(city_input, cuisine_list, min_rating, max_cost)

                if not results.empty:
                    st.success("Top matching restaurants:")
                    st.dataframe(results)
                else:
                    st.warning("No restaurants found matching your criteria.")
        except Exception as e:
            st.error(f"❌ Error in user_input: {e}")

if __name__ == '__main__':
    try:
        obj = SwiggyRecommender()
    except Exception as e:
        st.error(f"❌ Error in main(object creation): {e}")
        quit()
    try:
        obj.load_data()
        obj.user_input()
    except Exception as e:
        st.error(f"❌ Error in main(function calling): {e}")
