from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                            roc_curve, auc, mean_squared_error, log_loss, confusion_matrix, roc_auc_score) 
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import pickle
import os
import json
from wordcloud import WordCloud
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

# Define paths
DATA_PATH = '/home/yongjia/data'
LISTINGS_PATH = os.path.join(DATA_PATH, 'listings.csv')
CALENDAR_PATH = os.path.join(DATA_PATH, 'calendar.csv')
OUTPUT_PATH = '/home/yongjia/airflow/output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
REPORT_PATH = os.path.join(OUTPUT_PATH, 'reports')
VISUALIZATION_PATH = os.path.join(OUTPUT_PATH, 'visualizations')

# Create directories if they don't exist
for path in [OUTPUT_PATH, MODEL_PATH, REPORT_PATH, VISUALIZATION_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 4, 18)
}

# Define the DAG
dag = DAG(
    'airbnb_booking_probability_analysis',
    default_args=default_args,
    description='Airbnb booking probability and price elasticity analysis',
    schedule_interval=None,
    catchup=False
)


# def extract_data_from_bigquery(**context):
#     client = bigquery.Client()
    
#     listings_query = """
#     SELECT *
#     FROM `airbnb3107.airbnb.listings`
#     """
    
#     calendar_query = """
#     SELECT *
#     FROM `airbnb3107.airbnb.calendar`
#     """
    
#     reviews_query = """
#     SELECT *
#     FROM `airbnb3107.airbnb.reviews`
#     """
    
#     listings_df = client.query(listings_query).to_dataframe()
#     calendar_df = client.query(calendar_query).to_dataframe()
#     reviews_df = client.query(reviews_query).to_dataframe()
    
#     data_dir = 'data'
#     os.makedirs(data_dir, exist_ok=True)
    
#     listings_df.to_csv(f'{data_dir}/listings.csv', index=False)
#     calendar_df.to_csv(f'{data_dir}/calendar.csv', index=False)
#     reviews_df.to_csv(f'{data_dir}/reviews.csv', index=False)  # Fixed filename
    
#     context['ti'].xcom_push(key='data_dir', value=data_dir)
    
#     return {
#         'listings_shape': listings_df.shape,
#         'calendar_shape': calendar_df.shape,
#         'reviews_shape': reviews_df.shape
#     }


# Function to load and clean data
def load_and_clean_data(**kwargs): 
    try:
        # Handle listings data
        listings_df = pd.read_csv(LISTINGS_PATH)
        print(f"Original listings data shape: {listings_df.shape}")
        
        # Handle calendar data
        calendar_df = pd.read_csv(CALENDAR_PATH)
        print(f"Original calendar data shape: {calendar_df.shape}")
        
        # Basic data cleaning for listings
        # Convert price to numeric (remove $ and commas)
        if 'price' in listings_df.columns:
            listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)
        
        # Convert important columns to appropriate types
        numeric_cols = ['latitude', 'longitude', 'accommodates', 'bathrooms', 
                        'bedrooms', 'beds', 'minimum_nights', 'maximum_nights']
        
        for col in numeric_cols:
            if col in listings_df.columns:
                listings_df[col] = pd.to_numeric(listings_df[col], errors='coerce')
        
        # Drop rows with missing critical values
        critical_cols = ['id', 'latitude', 'longitude', 'price']
        critical_cols = [col for col in critical_cols if col in listings_df.columns]
        listings_df = listings_df.dropna(subset=critical_cols)
        
        # Basic data cleaning for calendar
        if 'price' in calendar_df.columns:
            calendar_df['price'] = calendar_df['price'].replace('[\$,]', '', regex=True).astype(float)
        
        # Convert available to boolean
        if 'available' in calendar_df.columns:
            calendar_df['available'] = calendar_df['available'].map({'t': True, 'f': False})
        
        # Convert date to datetime
        if 'date' in calendar_df.columns:
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            # Extract useful date features
            calendar_df['month'] = calendar_df['date'].dt.month
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek
            calendar_df['is_weekend'] = calendar_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Remove outliers from continuous data
        listings_df = remove_outliers(listings_df, columns=['price', 'accommodates', 'bedrooms', 'beds'])
        calendar_df = remove_outliers(calendar_df, columns=['price'])
        
        # Save cleaned data
        cleaned_listings_path = os.path.join(OUTPUT_PATH, 'cleaned_listings.csv')
        cleaned_calendar_path = os.path.join(OUTPUT_PATH, 'cleaned_calendar.csv')
        
        listings_df.to_csv(cleaned_listings_path, index=False)
        calendar_df.to_csv(cleaned_calendar_path, index=False)
        
        print(f"Cleaned listings data shape: {listings_df.shape}")
        print(f"Cleaned calendar data shape: {calendar_df.shape}")
        
        return {
            'listings_path': cleaned_listings_path,
            'calendar_path': cleaned_calendar_path
        }
    
    except Exception as e:
        print(f"Error in data loading and cleaning: {e}")
        raise

def process_reviews_data(**kwargs): 
    DATA_PATH = kwargs.get('data_path', os.environ.get('DATA_PATH', './data'))
    OUTPUT_PATH = kwargs.get('output_path', os.environ.get('OUTPUT_PATH', './output')) 
    VISUALIZATION_PATH = os.path.join(OUTPUT_PATH, 'visualizations')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'reports'), exist_ok=True)
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    # Load reviews data
    try: 
        reviews_path = os.path.join(DATA_PATH, 'reviews.csv')
        reviews_df = pd.read_csv(reviews_path)
        print(f"Original reviews data shape: {reviews_df.shape}")
        
        # Fix column name if necessary
        if 'listing_id' in reviews_df.columns:
            reviews_df.rename(columns={'list ing_id': 'listing_id'}, inplace=True)
        
        # Check for required columns
        required_columns = ['listing_id', 'comments']
        missing_columns = [col for col in required_columns if col not in reviews_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            # Try to map or find similar column names
            if 'listing_id' in missing_columns and any(col.lower().replace(' ', '').endswith('id') for col in reviews_df.columns):
                # Find potential listing ID columns
                potential_id_cols = [col for col in reviews_df.columns if col.lower().replace(' ', '').endswith('id')]
                if potential_id_cols:
                    first_id_col = potential_id_cols[0]
                    print(f"Using '{first_id_col}' as listing_id")
                    reviews_df.rename(columns={first_id_col: 'listing_id'}, inplace=True)
            
            # Check for comments or similar column
            if 'comments' in missing_columns:
                text_cols = [col for col in reviews_df.columns if col.lower() in ['comment', 'review', 'text', 'description']]
                if text_cols:
                    reviews_df.rename(columns={text_cols[0]: 'comments'}, inplace=True)
                    print(f"Using '{text_cols[0]}' as comments")
        
        # Check if there are any reviews with comments
        if len(reviews_df.dropna(subset=['comments'])) == 0:
            print("Warning: No reviews with comments found in the dataset")
            # Create empty sentiment file if no valid comments
            empty_df = pd.DataFrame(columns=['listing_id', 'signed_sentiment', 'review_count'])
            empty_df.to_csv(os.path.join(OUTPUT_PATH, 'listing_sentiment.csv'))
            return {
                'processed_reviews_path': None,
                'listing_sentiment_path': os.path.join(OUTPUT_PATH, 'listing_sentiment.csv')
            }
        
        # Filter out empty or missing comments
        reviews_df = reviews_df.dropna(subset=['comments'])
        reviews_df = reviews_df[reviews_df['comments'].str.strip() != '']
        
        # Ensure listing_id is properly formatted
        reviews_df['listing_id'] = pd.to_numeric(reviews_df['listing_id'], errors='coerce')
        reviews_df = reviews_df.dropna(subset=['listing_id'])
        reviews_df['listing_id'] = reviews_df['listing_id'].astype(int)
        
        from transformers import AutoTokenizer, pipeline
        
        # Load the tokenizer for the model
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        
        # Function to preprocess comments and handle long texts
        def preprocess_comment(text):
            if not isinstance(text, str):
                return ""
            # Clean the text - remove excessive whitespace
            text = ' '.join(text.split())
            return text
        
        # Apply preprocessing
        reviews_df['processed_comments'] = reviews_df['comments'].apply(preprocess_comment)
        
        # Import sentiment analysis model with truncation enabled
        sentiment_analyzer = pipeline(
            'sentiment-analysis', 
            model='distilbert-base-uncased-finetuned-sst-2-english', 
            tokenizer=tokenizer,
            truncation=True,  # Enable truncation for long sequences
            max_length=512    # Set maximum length
        )
        
        # Process in batches to avoid memory issues
        batch_size = 100
        sentiments = []
        error_indices = []
        
        total_batches = (len(reviews_df) + batch_size - 1) // batch_size
        
        # Track progress with tqdm if available
        try:
            batch_iterator = tqdm(range(0, len(reviews_df), batch_size), desc="Processing batches", total=total_batches)
        except:
            batch_iterator = range(0, len(reviews_df), batch_size)
            
        for i in batch_iterator:
            batch = reviews_df.iloc[i:i+batch_size]
            
            try:
                # Use processed comments for sentiment analysis
                batch_results = sentiment_analyzer(batch['processed_comments'].tolist())
                sentiments.extend(batch_results)
                print(f"Processed batch {i//batch_size + 1}/{total_batches}")
                
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Track which indices failed
                error_indices.extend(range(i, min(i+batch_size, len(reviews_df))))
                
                # Try processing one by one to salvage what we can
                for j in range(i, min(i+batch_size, len(reviews_df))):
                    try:
                        single_text = reviews_df.iloc[j]['processed_comments']
                        if isinstance(single_text, str) and single_text.strip():
                            result = sentiment_analyzer([single_text])[0]
                            sentiments.append(result)
                            # Remove from error indices if successful
                            if j in error_indices:
                                error_indices.remove(j)
                    except Exception as single_error:
                        print(f"Failed to process comment at index {j}: {str(single_error)}")
        
        # Check if we have any successful results before proceeding
        if not sentiments:
            print("Error: No comments could be processed for sentiment analysis")
            empty_df = pd.DataFrame(columns=['listing_id', 'signed_sentiment', 'review_count'])
            empty_df.to_csv(os.path.join(OUTPUT_PATH, 'listing_sentiment.csv'))
            return {
                'processed_reviews_path': None,
                'listing_sentiment_path': os.path.join(OUTPUT_PATH, 'listing_sentiment.csv')
            }
        
        # Create a new dataframe excluding error rows
        success_indices = [i for i in range(len(reviews_df)) if i not in error_indices]
        success_df = reviews_df.iloc[success_indices].copy()
        
        # Now we need to make sure our sentiments list matches our dataframe
        if len(success_df) != len(sentiments):
            print(f"Warning: Number of processed reviews ({len(success_df)}) doesn't match sentiment results ({len(sentiments)})")
            # Adjust if needed to ensure proper alignment
            min_len = min(len(success_df), len(sentiments))
            success_df = success_df.iloc[:min_len]
            sentiments = sentiments[:min_len]
        
        # Add sentiment scores to the dataframe
        success_df['sentiment_label'] = [item['label'] for item in sentiments]
        success_df['sentiment_score'] = [item['score'] for item in sentiments]
        
        # Convert sentiment labels to numeric (-1 for NEGATIVE, 1 for POSITIVE)
        success_df['sentiment_value'] = success_df['sentiment_label'].map({'NEGATIVE': -1, 'POSITIVE': 1})
        
        # Multiply score by value to get signed sentiment score
        success_df['signed_sentiment'] = success_df['sentiment_value'] * success_df['sentiment_score']
        
        # Aggregate sentiment by listing_id
        listing_sentiment = success_df.groupby('listing_id').agg({
            'signed_sentiment': 'mean',
            'listing_id': 'count'  # Count by listing_id for review count
        })
        
        # Rename the count column to review_count
        listing_sentiment.rename(columns={'listing_id': 'review_count'}, inplace=True)
        
        # Reset index to make listing_id a column
        listing_sentiment = listing_sentiment.reset_index()
        
        print(f"Generated sentiment for {len(listing_sentiment)} listings")
        print(f"Successfully processed {len(success_df)} out of {len(reviews_df)} reviews")
        
        # Save processed reviews data
        processed_reviews_path = os.path.join(OUTPUT_PATH, 'processed_reviews.csv')
        success_df.to_csv(processed_reviews_path, index=False)
        
        # Save aggregated sentiment by listing
        listing_sentiment_path = os.path.join(OUTPUT_PATH, 'listing_sentiment.csv')
        listing_sentiment.to_csv(listing_sentiment_path, index=False)
        
        # Create sentiment visualization
        plt.figure(figsize=(10, 6))
        plt.hist(listing_sentiment['signed_sentiment'], bins=30)
        plt.title('Distribution of Average Sentiment Scores by Listing')
        plt.xlabel('Sentiment Score (-1 to 1)')
        plt.ylabel('Count of Listings')
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'sentiment_distribution.png'))
        plt.close()
        
        # Also create visualization for review counts
        plt.figure(figsize=(10, 6))
        plt.hist(listing_sentiment['review_count'], bins=30)
        plt.title('Distribution of Review Counts by Listing')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Count of Listings')
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'review_count_distribution.png'))
        plt.close()
        
        return {
            'processed_reviews_path': processed_reviews_path,
            'listing_sentiment_path': listing_sentiment_path
        }
    
    except Exception as e:
        print(f"Error in processing reviews data: {e}")
        traceback.print_exc()
        
        # Create empty sentiment file if reviews processing fails
        empty_df = pd.DataFrame(columns=['listing_id', 'signed_sentiment', 'review_count'])
        empty_df.to_csv(os.path.join(OUTPUT_PATH, 'listing_sentiment.csv'))
        return {
            'processed_reviews_path': None,
            'listing_sentiment_path': os.path.join(OUTPUT_PATH, 'listing_sentiment.csv')
        }
    
def remove_outliers(df, columns=['price']):
    """Remove outliers from continuous data columns"""
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            # Calculate Q1, Q3 and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers (1.5 IQR method)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            after_count = len(df_clean)
            
            print(f"Removed {before_count - after_count} outliers from {col}")
    
    return df_clean

def json_serializer(obj):
    if isinstance(obj, (np.integer, np.uint32)):
        return int(obj)
    elif isinstance(obj, (np.float, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Series)):
        return {str(k): v for k, v in obj.items()}
    return str(obj)

# Function for exploratory data analysis and visualization
def exploratory_analysis(**kwargs):
    ti = kwargs['ti']
    file_paths = ti.xcom_pull(task_ids='load_and_clean_data')
    
    # Load data
    listings_df = pd.read_csv(file_paths['listings_path']) 
    calendar_df = pd.read_csv(file_paths['calendar_path'])
    
    # Convert date back to datetime
    if 'date' in calendar_df.columns:
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("muted")
    
    # Initialize results dictionary to store insights
    insights = {
        "listings_count": len(listings_df),
        "price_stats": {},
        "occupancy_stats": {},
        "top_neighborhoods": [],
        "amenities_analysis": {},
        "correlations": {}
    }
    
    # 1. Price Distribution Analysis
    plt.figure(figsize=(12, 8))
    
    # Boxplot of prices
    plt.subplot(2, 2, 1)
    sns.boxplot(y=listings_df['price'])
    plt.title('Price Distribution (Box Plot)')
    plt.ylabel('Price ($)')
    
    # Histogram of prices
    plt.subplot(2, 2, 2)
    sns.histplot(listings_df['price'].clip(upper=listings_df['price'].quantile(0.95)), kde=True)
    plt.title('Price Distribution (Histogram - 95th percentile)')
    plt.xlabel('Price ($)')
    
    # Log-transformed price distribution
    plt.subplot(2, 2, 3)
    sns.histplot(np.log1p(listings_df['price']), kde=True)
    plt.title('Log-transformed Price Distribution')
    plt.xlabel('Log(Price + 1)')
    
    # Price by accommodation capacity
    plt.subplot(2, 2, 4)
    if 'accommodates' in listings_df.columns:
        sns.boxplot(x='accommodates', y='price', data=listings_df[listings_df['accommodates'] <= 10])
        plt.title('Price by Accommodation Capacity')
        plt.xlabel('Accommodates')
        plt.ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_PATH, 'price_distribution_analysis.png'))
    plt.close()
    
    # Store price statistics in insights
    insights["price_stats"] = {
        "mean": listings_df['price'].mean(),
        "median": listings_df['price'].median(),
        "min": listings_df['price'].min(),
        "max": listings_df['price'].max(),
        "std": listings_df['price'].std()
    }
    
    # 2. Geographic Analysis
    if all(col in listings_df.columns for col in ['latitude', 'longitude']):
        plt.figure(figsize=(14, 10))
        
        # Remove outliers for visualization
        lat_bounds = listings_df['latitude'].quantile([0.01, 0.99]).values
        lon_bounds = listings_df['longitude'].quantile([0.01, 0.99]).values
        
        geo_df = listings_df[
            (listings_df['latitude'] >= lat_bounds[0]) & 
            (listings_df['latitude'] <= lat_bounds[1]) &
            (listings_df['longitude'] >= lon_bounds[0]) & 
            (listings_df['longitude'] <= lon_bounds[1])
        ]
        
        # Price heatmap by location
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            geo_df['longitude'], 
            geo_df['latitude'], 
            c=geo_df['price'], 
            cmap='viridis', 
            alpha=0.6, 
            s=50
        )
        plt.colorbar(scatter, label='Price ($)')
        plt.title('Listing Prices by Location')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Density map of listings
        plt.subplot(1, 2, 2)
        sns.kdeplot(
            x=geo_df['longitude'],
            y=geo_df['latitude'],
            cmap='Reds', 
            fill=True,
            alpha=0.7
        )
        plt.title('Listing Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'geographic_analysis.png'))
        plt.close()
    
    # 3. Neighborhood Analysis
    if 'neighbourhood' in listings_df.columns or 'neighbourhood_cleansed' in listings_df.columns:
        neighbourhood_col = 'neighbourhood' if 'neighbourhood' in listings_df.columns else 'neighbourhood_cleansed'
        
        # Count listings by neighborhood
        neighborhood_counts = listings_df[neighbourhood_col].value_counts().head(15)
        
        # Average price by neighborhood
        neighborhood_prices = listings_df.groupby(neighbourhood_col)['price'].mean().sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(18, 8))
        
        # Neighborhood Listing Counts
        plt.subplot(1, 2, 1)
        sns.barplot(x=neighborhood_counts.values, y=neighborhood_counts.index)
        plt.title('Top 15 Neighborhoods by Listing Count')
        plt.xlabel('Number of Listings')
        
        # Neighborhood Average Prices
        plt.subplot(1, 2, 2)
        sns.barplot(x=neighborhood_prices.values, y=neighborhood_prices.index)
        plt.title('Top 15 Neighborhoods by Average Price')
        plt.xlabel('Average Price ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'neighborhood_analysis.png'))
        plt.close()
        
        # Store top neighborhoods by listing count and average price
        insights["top_neighborhoods"] = {
            "by_count": {str(k): v for k, v in neighborhood_counts.to_dict().items()},
            "by_price": {str(k): v for k, v in neighborhood_prices.to_dict().items()}
        }
 
    # 4. Property Characteristics Analysis
    plt.figure(figsize=(18, 12))
    
    # Price by Room Type
    if 'room_type' in listings_df.columns:
        plt.subplot(2, 2, 1)
        sns.boxplot(x='room_type', y='price', data=listings_df)
        plt.title('Price by Room Type')
        plt.xticks(rotation=45)
        plt.xlabel('Room Type')
        plt.ylabel('Price ($)')
    
    # Price by Property Type (top 10)
    if 'property_type' in listings_df.columns:
        plt.subplot(2, 2, 2)
        # Get top 10 property types by count
        top_property_types = listings_df['property_type'].value_counts().head(10).index
        property_type_df = listings_df[listings_df['property_type'].isin(top_property_types)]
        
        sns.boxplot(x='property_type', y='price', data=property_type_df)
        plt.title('Price by Property Type (Top 10)')
        plt.xticks(rotation=90)
        plt.xlabel('Property Type')
        plt.ylabel('Price ($)')
    
    # Price by Number of Bedrooms
    if 'bedrooms' in listings_df.columns:
        plt.subplot(2, 2, 3)
        # Filter to remove extreme outliers
        bedroom_df = listings_df[listings_df['bedrooms'] <= 6]
        sns.boxplot(x='bedrooms', y='price', data=bedroom_df)
        plt.title('Price by Number of Bedrooms')
        plt.xlabel('Bedrooms')
        plt.ylabel('Price ($)')
    
    # Price by Number of Beds
    if 'beds' in listings_df.columns:
        plt.subplot(2, 2, 4)
        # Filter to remove extreme outliers
        beds_df = listings_df[listings_df['beds'] <= 8]
        sns.boxplot(x='beds', y='price', data=beds_df)
        plt.title('Price by Number of Beds')
        plt.xlabel('Beds')
        plt.ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_PATH, 'property_characteristics.png'))
    plt.close()
    
    # 5. Amenities Analysis
    if 'amenities' in listings_df.columns:
        # Parse amenities from JSON string format
        def extract_amenities(amenities_str):
            try:
                if isinstance(amenities_str, str):
                    # Clean the string and convert to list
                    amenities_str = amenities_str.replace('\"', '"').replace('"[', '[').replace(']"', ']')
                    amenities_list = json.loads(amenities_str.replace("'", '"'))
                    return amenities_list
                return []
            except:
                try:
                    # Alternative parsing if JSON loading fails
                    if isinstance(amenities_str, str):
                        return [a.strip('" ') for a in amenities_str.strip('[]').split(',')]
                    return []
                except:
                    return []
        
        # Extract amenities and create new columns
        amenities_lists = listings_df['amenities'].apply(extract_amenities)
        
        # Count amenities frequency
        all_amenities = []
        for amenities in amenities_lists:
            all_amenities.extend(amenities)
        
        amenities_counts = pd.Series(all_amenities).value_counts()
        
        # Create dummy variables for top amenities
        top_amenities = amenities_counts.head(20).index
        
        for amenity in top_amenities:
            listings_df[f'has_{amenity.lower().replace(" ", "_")}'] = amenities_lists.apply(
                lambda x: 1 if amenity in x else 0
            )
        
        # Amenities frequency chart
        plt.figure(figsize=(14, 8))
        sns.barplot(x=amenities_counts.head(20).values, y=amenities_counts.head(20).index)
        plt.title('Top 20 Most Common Amenities')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'amenities_frequency.png'))
        plt.close()
        
        # Price comparison for top amenities
        plt.figure(figsize=(16, 10))
        
        # Create subplots for each amenity's price impact
        for i, amenity in enumerate(top_amenities[:12], 1):
            col_name = f'has_{amenity.lower().replace(" ", "_")}'
            if col_name in listings_df.columns:
                plt.subplot(3, 4, i)
                sns.boxplot(x=col_name, y='price', data=listings_df)
                plt.title(f'Price by Presence of {amenity}')
                plt.xticks([0, 1], ['No', 'Yes'])
                plt.ylabel('Price ($)' if i % 4 == 1 else '')
                plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'amenities_price_impact.png'))
        plt.close()
        
        # Store amenities analysis in insights
        insights["amenities_analysis"] = {
            "top_amenities": {str(k): int(v) for k, v in amenities_counts.head(20).items()},
            "amenities_count_stats": {
                "mean": amenities_lists.apply(len).mean(),
                "median": amenities_lists.apply(len).median(),
                "max": amenities_lists.apply(len).max()
            }
        }
        
        # Word cloud of amenities
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(' '.join(all_amenities))
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'amenities_wordcloud.png'))
        plt.close()
    
    # 6. Calendar Analysis (Availability and Pricing Over Time)
    if 'date' in calendar_df.columns:
        # Weekly average price
        weekly_prices = calendar_df.groupby(calendar_df['date'].dt.isocalendar().week)['price'].mean()
        
        # Monthly average price
        monthly_prices = calendar_df.groupby(calendar_df['date'].dt.month)['price'].mean()
        
        # Availability over time
        if 'available' in calendar_df.columns:
            # Weekly availability rate
            weekly_availability = calendar_df.groupby(calendar_df['date'].dt.isocalendar().week)['available'].mean()
            
            # Monthly availability rate
            monthly_availability = calendar_df.groupby(calendar_df['date'].dt.month)['available'].mean()
            
            plt.figure(figsize=(16, 12))
            
            # Weekly price and availability
            plt.subplot(2, 1, 1)
            ax1 = plt.gca()
            ax1.set_ylabel('Average Price ($)', color='tab:blue')
            ax1.plot(weekly_prices.index, weekly_prices.values, 'o-', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Availability Rate', color='tab:red')
            ax2.plot(weekly_availability.index, weekly_availability.values, 'o-', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title('Weekly Average Price and Availability Rate')
            plt.xlabel('Week of Year')
            
            # Monthly price and availability
            plt.subplot(2, 1, 2)
            ax1 = plt.gca()
            ax1.set_ylabel('Average Price ($)', color='tab:blue')
            ax1.plot(monthly_prices.index, monthly_prices.values, 'o-', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Availability Rate', color='tab:red')
            ax2.plot(monthly_availability.index, monthly_availability.values, 'o-', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title('Monthly Average Price and Availability Rate')
            plt.xlabel('Month')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_PATH, 'calendar_analysis.png'))
            plt.close()
            
            # Store occupancy stats in insights
            insights["occupancy_stats"] = {
                "average_availability_rate": calendar_df['available'].mean(),
                "weekly_availability": {int(k): float(v) for k, v in weekly_availability.items()},
                "monthly_availability": {int(k): float(v) for k, v in monthly_availability.items()},
            }
            
            # Weekday vs Weekend Analysis
            calendar_df['is_weekend'] = calendar_df['date'].dt.dayofweek >= 5
            weekday_weekend_price = calendar_df.groupby('is_weekend')['price'].mean()
            weekday_weekend_availability = calendar_df.groupby('is_weekend')['available'].mean()
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=['Weekday', 'Weekend'], y=weekday_weekend_price.values)
            plt.title('Average Price: Weekday vs Weekend')
            plt.ylabel('Average Price ($)')
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=['Weekday', 'Weekend'], y=weekday_weekend_availability.values)
            plt.title('Availability Rate: Weekday vs Weekend')
            plt.ylabel('Availability Rate')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_PATH, 'weekday_weekend_analysis.png'))
            plt.close()
    
    # 7. Review Scores Analysis
    review_score_cols = [col for col in listings_df.columns if col.startswith('review_scores_')]
    
    if review_score_cols:
        plt.figure(figsize=(14, 8))
        
        # Distribution of overall rating
        if 'review_scores_rating' in listings_df.columns:
            plt.subplot(2, 2, 1)
            sns.histplot(listings_df['review_scores_rating'].dropna(), kde=True)
            plt.title('Distribution of Overall Rating')
            plt.xlabel('Rating')
        
        # Average subscores
        subscores = [col for col in review_score_cols if col != 'review_scores_rating']
        if subscores:
            plt.subplot(2, 2, 2)
            subscore_means = listings_df[subscores].mean().sort_values()
            sns.barplot(x=subscore_means.values, y=subscore_means.index)
            plt.title('Average Review Subscores')
            plt.xlabel('Average Score')
        
        # Correlation between price and ratings
        if 'review_scores_rating' in listings_df.columns:
            plt.subplot(2, 2, 3)
            sns.scatterplot(
                x='price', 
                y='review_scores_rating', 
                data=listings_df[listings_df['price'] <= listings_df['price'].quantile(0.95)]
            )
            plt.title('Price vs. Overall Rating')
            plt.xlabel('Price ($)')
            plt.ylabel('Rating')
        
        # Ratings over time (if first_review date is available)
        if 'first_review' in listings_df.columns and 'review_scores_rating' in listings_df.columns:
            try:
                listings_df['first_review'] = pd.to_datetime(listings_df['first_review'])
                listings_df['review_year'] = listings_df['first_review'].dt.year
                
                plt.subplot(2, 2, 4)
                yearly_ratings = listings_df.groupby('review_year')['review_scores_rating'].mean()
                sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values)
                plt.title('Average Rating by First Review Year')
                plt.xlabel('Year')
                plt.ylabel('Average Rating')
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'review_scores_analysis.png'))
        plt.close()
    
    # 8. Correlation Analysis
    # Select numeric columns for correlation analysis
    numeric_cols = listings_df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter for relevant numeric columns
    relevant_cols = [
        'price', 'minimum_nights', 'maximum_nights', 'accommodates', 
        'bedrooms', 'beds', 'bathrooms', 'number_of_reviews'
    ]
    
    corr_cols = [col for col in relevant_cols if col in numeric_cols]
    
    if len(corr_cols) > 1:
        corr_matrix = listings_df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix of Listing Features')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'correlation_analysis.png'))
        plt.close()
        
        # Store correlations with price in insights
        if 'price' in corr_cols:
            insights["correlations"] = {str(k): float(v) for k, v in corr_matrix['price'].sort_values(ascending=False).items()}
    
    # Save insights to JSON file
    with open(os.path.join(OUTPUT_PATH, 'eda_insights.json'), 'w') as f:
        json.dump(insights, f, indent=4, default=json_serializer)
    
    return {
        'listings_path': file_paths['listings_path'],
        'calendar_path': file_paths['calendar_path'],
        'viz_path': VISUALIZATION_PATH
    }

def feature_engineering(**kwargs):
    ti = kwargs['ti']
    file_paths = ti.xcom_pull(task_ids='load_and_clean_data')
    
    # Check both ways of accessing the sentiment path
    sentiment_paths = ti.xcom_pull(task_ids='process_reviews_data')
    if isinstance(sentiment_paths, dict) and 'listing_sentiment_path' in sentiment_paths:
        sentiment_path = sentiment_paths['listing_sentiment_path']
    else:
        # Try direct pull of the specific key
        sentiment_path = ti.xcom_pull(task_ids='process_reviews_data', key='listing_sentiment_path')
        
    print(f"Sentiment path retrieved: {sentiment_path}")
    listings_df = pd.read_csv(file_paths['listings_path'])
    calendar_df = pd.read_csv(file_paths['calendar_path'])
    
    # Convert date back to datetime
    if 'date' in calendar_df.columns:
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # Make sure we have matching column names for the join
    listings_id_col = 'id' if 'id' in listings_df.columns else 'listing_id'
    calendar_id_col = 'listing_id' if 'listing_id' in calendar_df.columns else 'id'
    
    # Create a copy of dataframes with consistent id column names
    listings_temp = listings_df.copy()
    calendar_temp = calendar_df.copy()
    
    if listings_id_col != 'listing_id':
        listings_temp['listing_id'] = listings_temp[listings_id_col]
    
    if calendar_id_col != 'listing_id':
        calendar_temp['listing_id'] = calendar_temp[calendar_id_col]
    
    # Before merging, check which price columns exist
    print(f"Calendar columns: {calendar_temp.columns.tolist()}")
    print(f"Listings columns: {listings_temp.columns.tolist()}")
    
    # Temporarily rename price columns to avoid conflicts during merge
    if 'price' in calendar_temp.columns:
        calendar_temp.rename(columns={'price': 'calendar_price'}, inplace=True)
    
    if 'price' in listings_temp.columns:
        listings_temp.rename(columns={'price': 'listing_price'}, inplace=True)
    
    # Merge the data
    try:
        merged_df = pd.merge(calendar_temp, listings_temp, on='listing_id', how='inner')
        print(f"Merged data shape: {merged_df.shape}")
    except Exception as e:
        print(f"Error in merging: {e}")
        merged_df = calendar_temp.copy()
    
    # Create the final price column, prioritizing calendar price
    if 'calendar_price' in merged_df.columns:
        # Create or restore the 'price' column
        merged_df['price'] = merged_df['calendar_price']
    elif 'listing_price' in merged_df.columns:
        merged_df['price'] = merged_df['listing_price']
    elif 'adjusted_price' in merged_df.columns:
        merged_df['price'] = merged_df['adjusted_price']
    
    # Drop rows without price
    print(f"Before dropping rows without price: {merged_df.shape}")
    merged_df = merged_df.dropna(subset=['price'])
    print(f"After dropping rows without price: {merged_df.shape}")
    
    # Convert price to numeric
    merged_df['price'] = pd.to_numeric(merged_df['price'], errors='coerce')
    
    # Feature Engineering
    # 1. Create booking_status (inverse of available)
    if 'available' in merged_df.columns:
        merged_df['booking_status'] = ~merged_df['available']
        merged_df['booking_status'] = merged_df['booking_status'].astype(int)
    
    # 2. Create price features
    # Log transform price to handle skewness
    merged_df['log_price'] = np.log1p(merged_df['price'])
    
    # Price buckets
    merged_df['price_bucket'] = pd.qcut(merged_df['price'], q=5, labels=False)
    
    # 3. Create seasonality features
    if 'date' in merged_df.columns:
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
        merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)
        merged_df['season'] = pd.cut(merged_df['month'], 
                                     bins=[0, 3, 6, 9, 12], 
                                     labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # 4. Create interaction features
    if 'beds' in merged_df.columns and 'price' in merged_df.columns:
        merged_df['price_per_bed'] = merged_df['price'] / merged_df['beds'].replace(0, 1)
    
    if 'accommodates' in merged_df.columns and 'price' in merged_df.columns:
        merged_df['price_per_person'] = merged_df['price'] / merged_df['accommodates'].replace(0, 1)
    
    # 5. Room type and property features (if available)
    categorical_cols = ['room_type', 'property_type', 'season']
    cat_cols_available = [col for col in categorical_cols if col in merged_df.columns]
    
    if cat_cols_available:
        # One-hot encode categorical features
        for col in cat_cols_available:
            dummies = pd.get_dummies(merged_df[col], prefix=col, drop_first=True)
            merged_df = pd.concat([merged_df, dummies], axis=1)
    
    # 6. Additional features for price elasticity analysis
    # Add price difference from neighborhood mean
    if 'neighbourhood' in merged_df.columns and 'price' in merged_df.columns:
        neighbourhood_avg_price = merged_df.groupby('neighbourhood')['price'].transform('mean')
        merged_df['price_diff_from_neighbourhood_avg'] = merged_df['price'] - neighbourhood_avg_price
        merged_df['price_ratio_to_neighbourhood_avg'] = merged_df['price'] / neighbourhood_avg_price
    
    # 7. Add booking pattern features
    if 'listing_id' in merged_df.columns and 'booking_status' in merged_df.columns:
        # Booking rate by listing
        listing_booking_rate = merged_df.groupby('listing_id')['booking_status'].transform('mean')
        merged_df['listing_historical_booking_rate'] = listing_booking_rate
    
    # Drop rows with NaN in critical columns for modeling
    critical_modeling_cols = ['price']
    if 'booking_status' in merged_df.columns:
        critical_modeling_cols.append('booking_status')

    # Process amenities into features
    if 'amenities' in listings_df.columns:
        # Convert JSON string to list
        listings_df['amenities_list'] = listings_df['amenities'].apply(
            lambda x: json.loads(x.replace('"', '"').replace('"', '"')) if isinstance(x, str) else []
        )
        
        # Create binary features for important amenities
        key_amenities = [
            'Wifi', 'Kitchen', 'Free parking', 'Pool', 'Hot tub', 
            'Air conditioning', 'Heating', 'Washer', 'Dryer',
            'Body soap', 'Oven', 'Stainless steel oven', 
            'Portable air conditioning', 'Cooking basics', 
            'First aid kit', 'Exercise equipment', 'Shower gel'
        ]

        for amenity in key_amenities:
            listings_df[f'has_{amenity.lower().replace(" ", "_")}'] = listings_df['amenities_list'].apply(
                lambda x: any(amenity.lower() in item.lower() for item in x) if isinstance(x, list) else False
            )
    
    # Count total amenities as a feature
    listings_df['amenities_count'] = listings_df['amenities_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)    
    # Process review scores
    review_score_cols = [col for col in listings_df.columns if col.startswith('review_scores_')]
    
    # Create an overall review score (average of all review scores)
    if review_score_cols:
        listings_df['avg_review_score'] = listings_df[review_score_cols].mean(axis=1) 
    
    # Create supply/demand metrics by geography
    if 'neighbourhood' in listings_df.columns:
        neighborhood_counts = listings_df.groupby('neighbourhood').size()
        listings_df['area_supply'] = listings_df['neighbourhood'].map(neighborhood_counts)
    
    # Additional time-based features for calendar data
    if 'date' in calendar_df.columns:
        # Season based on Northern Hemisphere
        calendar_df['season'] = pd.cut(
            calendar_df['date'].dt.month,
            bins=[0, 2, 5, 8, 11, 12],   
            labels=['Winter (Jan–Feb)', 'Spring', 'Summer', 'Fall', 'Winter (Dec)'],
            ordered=True,
            include_lowest=True
        )

        # Is holiday season (December-January)
        calendar_df['is_holiday_season'] = calendar_df['date'].dt.month.isin([12, 1]).astype(int)
        
        # Days from today
        today = pd.Timestamp.now()
        calendar_df['days_from_today'] = (calendar_df['date'] - today).dt.days
        
        # Booking lead time buckets
        calendar_df['lead_time_bucket'] = pd.cut(
            calendar_df['days_from_today'],
            bins=[-float('inf'), 7, 30, 90, float('inf')],
            labels=['Last minute', 'Short term', 'Medium term', 'Long term']
        )
    
    try:
        if sentiment_path and os.path.exists(sentiment_path):
            sentiment_df = pd.read_csv(sentiment_path)
            has_sentiment = True
            print(f"Loaded sentiment data with shape: {sentiment_df.shape}")
        else:
            has_sentiment = False
            print(f"Sentiment file not found at path: {sentiment_path}")
    except Exception as e:
        has_sentiment = False
        print(f"Error loading sentiment data: {e}")
        
    if has_sentiment:
        merged_df = pd.merge(merged_df, sentiment_df, on='listing_id', how='left')
        # Fill missing sentiment with neutral value (0)
        merged_df['signed_sentiment'] = merged_df['signed_sentiment'].fillna(0)
        merged_df['review_count'] = merged_df['review_count'].fillna(0)
        
        # Create sentiment buckets
        sentiment_bins = [-0.99, -0.5, -0.1, 0.1, 0.5, 0.99]
        merged_df['sentiment_bucket'] = pd.cut(
            merged_df['signed_sentiment'].clip(-0.99, 0.99),
            bins=sentiment_bins,
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        )
        
        # Create sentiment visualization by room type
        if 'room_type' in merged_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='room_type', y='signed_sentiment', data=merged_df)
            plt.title('Sentiment by Room Type')
            plt.savefig(os.path.join(VISUALIZATION_PATH, 'sentiment_by_room_type.png'))
            plt.close()
        
        # Visualize relationship between sentiment and price
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='signed_sentiment', y='price', data=merged_df.sample(min(5000, len(merged_df))))
        plt.title('Price vs Sentiment')
        plt.savefig(os.path.join(VISUALIZATION_PATH, 'price_vs_sentiment.png'))
        plt.close()

    merged_df = merged_df.dropna(subset=critical_modeling_cols)
    
    # Save the processed data
    processed_data_path = os.path.join(OUTPUT_PATH, 'processed_data.csv')
    merged_df.to_csv(processed_data_path, index=False)
    
    print(f"Processed data shape: {merged_df.shape}")
    return {'processed_data_path': processed_data_path}

# Function to calculated elasticity, to add anywhere u want use to maybe create more visuals 
def calculate_price_elasticity(df, model, feature_names):
    """Calculate price elasticity across different segments"""
    # Create segments for analysis
    segments = []
    
    # Add seasonality if available
    if 'season' in df.columns:
        seasons = df['season'].unique()
        segments.extend([('season', season) for season in seasons])
    
    # Add weekend/weekday if available
    if 'is_weekend' in df.columns:
        segments.extend([('is_weekend', 0), ('is_weekend', 1)])
    
    # Add property types if available
    property_cols = [col for col in df.columns if col.startswith('property_type_')]
    for col in property_cols:
        segments.append((col, 1))
    
    # Base elasticity (overall)
    elasticity_results = []
    
    # Function to compute elasticity for a segment
    def compute_segment_elasticity(segment_df, segment_name):
        if len(segment_df) < 50:  # Skip if not enough data
            return None
        
        X = segment_df[feature_names].copy()
        
        # Check if price is in features
        if 'price' not in X.columns:
            return None
        
        # Create price variations for elasticity calculation
        price_variations = np.linspace(0.8, 1.2, 5)  # -20% to +20%
        base_prices = segment_df['price'].values
        
        booking_probs = []
        for pv in price_variations:
            # Copy the features and modify price
            X_modified = X.copy()
            X_modified['price'] = base_prices * pv
            
            # If log_price is a feature, update it too
            if 'log_price' in X_modified.columns:
                X_modified['log_price'] = np.log1p(X_modified['price'])
            
            # If price_per_person is a feature, update it
            if 'price_per_person' in X_modified.columns and 'accommodates' in segment_df.columns:
                X_modified['price_per_person'] = X_modified['price'] / segment_df['accommodates'].replace(0, 1)
            
            # If price_per_bed is a feature, update it
            if 'price_per_bed' in X_modified.columns and 'beds' in segment_df.columns:
                X_modified['price_per_bed'] = X_modified['price'] / segment_df['beds'].replace(0, 1)
            
            # Predict booking probability
            if hasattr(model, 'predict_proba'):
                booking_prob = model.predict_proba(X_modified)[:, 1].mean()
            else:
                booking_prob = model.predict(X_modified).mean()
            
            booking_probs.append(float(booking_prob))
        
        # Calculate elasticity using middle points
        price_change_pct = (price_variations[3] - price_variations[1]) / price_variations[1]
        prob_change_pct = (booking_probs[3] - booking_probs[1]) / booking_probs[1] if booking_probs[1] > 0 else 0
        
        elasticity = prob_change_pct / price_change_pct if price_change_pct != 0 else 0
        
        # Convert price variations to standard Python types
        price_variations_list = [float(pv) for pv in price_variations]
        
        return {
            'segment': segment_name,
            'elasticity': float(elasticity),
            'avg_booking_prob': float(np.mean(booking_probs)),
            'price_response': list(zip(price_variations_list, booking_probs))
        }
    
    # Calculate overall elasticity
    overall_elasticity = compute_segment_elasticity(df, 'Overall')
    if overall_elasticity:
        elasticity_results.append(overall_elasticity)
    
    # Calculate elasticity for each segment
    for seg_col, seg_val in segments:
        segment_df = df[df[seg_col] == seg_val]
        segment_name = f"{seg_col}_{seg_val}"
        segment_elasticity = compute_segment_elasticity(segment_df, segment_name)
        if segment_elasticity:
            elasticity_results.append(segment_elasticity)
    
    return elasticity_results

def make_serializable(obj):
    """Convert NumPy types to standard Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    else:
        return obj 

class MultiMethodFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None, selected_features=None):
        self.feature_names = feature_names
        self.selected_features = selected_features
        
    def fit(self, X, y=None):
        # Store internal copies converted to lists
        self.feature_names_ = list(self.feature_names) if self.feature_names is not None else []
        self.selected_features_ = list(self.selected_features) if self.selected_features is not None else []
        
        # Create selection mask
        self.selection_mask_ = np.array([name in self.selected_features_ for name in self.feature_names_])
        return self
            
    def transform(self, X):
        check_is_fitted(self, 'selection_mask_')
        return X[:, self.selection_mask_]
            
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selection_mask_')
        return np.array(self.feature_names_)[self.selection_mask_]

class PCASelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        
    def fit(self, X, y=None):
        self.pca.fit(X)
        # Store explained variance info
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.n_components_selected_ = self.pca.n_components_
        return self
            
    def transform(self, X):
        return self.pca.transform(X)
    
def train_models(**kwargs):
    ti = kwargs['ti']
    processed_data_path = ti.xcom_pull(task_ids='feature_engineering')['processed_data_path']
    
    # Load processed data
    df = pd.read_csv(processed_data_path)
    print(f"Training advanced models with data of shape: {df.shape}")
    
    # Create output folder for model results
    models_path = os.path.join(OUTPUT_PATH, 'models')
    os.makedirs(models_path, exist_ok=True)
    
    # Model results storage
    model_results = {
        "booking_probability": {},
        "price_prediction": {},
        "feature_importance": {}
    }
    
    # Define features for booking probability prediction
    booking_features = [
        'host_is_superhost', 'security_deposit', 'security_deposit',
        'is_location_exact', 'host_identity_verified', 'host_response_time',
        
        # Basic listing attributes
        'price', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'minimum_nights', 'maximum_nights', 'is_weekend', 'month',
        
        # Amenity features
        'has_wifi', 'has_kitchen', 'has_free_parking', 'has_pool', 
        'has_hot_tub', 'has_air_conditioning', 'amenities_count',
        
        # Review features
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'avg_review_score', 'signed_sentiment'
        
        # Derived features
        'price_per_person', 'price_per_bed', 'price_per_rating_point',
        'area_supply', 'is_holiday_season', 'lead_time_bucket'
    ]
    
    # Only use features that exist in the dataframe
    booking_features = [f for f in booking_features if f in df.columns]
    print(f"Missing values before preprocessing: {df[booking_features].isnull().sum().sum()}")

    # Optional: Print columns with missing values
    missing_cols = df[booking_features].columns[df[booking_features].isnull().any()].tolist()
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        print(df[missing_cols].isnull().sum())

    df = df.dropna(subset=booking_features)
    
    # Add categorical features
    categorical_features = [
        col for col in df.columns
        if col.startswith('room_type_') or col.startswith('property_type_') or 
        col.startswith('season_') or col == 'lead_time_bucket'
    ]
    booking_features.extend(categorical_features)
    all_features = booking_features

    feature_names_path = os.path.join(models_path, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(all_features, f)
    
    # Prepare data for booking probability model
    if 'booking_status' in df.columns:
        X = df[booking_features].copy()
        y = df['booking_status']
        
        # Handle categorical features
        cat_cols = [col for col in booking_features if 
                   df[col].dtype == 'object' or df[col].dtype == 'category']
        num_cols = [col for col in booking_features if col not in cat_cols]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), cat_cols)
            ],
            remainder='passthrough'
        )
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Neural Network': MLPClassifier(max_iter=500, random_state=42)
        }
        
        best_models = {}
        
        # Feature selection approaches
        
        # 1. Preprocess the data first for feature selection
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        
        # 2. Multiple feature selection methods
        feature_selection_results = {}
        
        # 2.1. Random Forest Importance
        print("Calculating Random Forest feature importance...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X_train_processed, y_train)
        rf_importances = rf_selector.feature_importances_
        rf_indices = np.argsort(rf_importances)[::-1]
        
        feature_selection_results['random_forest'] = {
            'feature_names': feature_names[rf_indices],
            'importance_values': rf_importances[rf_indices]
        }
        
        # Save RF importance visualization
        plt.figure(figsize=(12, 8))
        plt.barh(feature_names[rf_indices][:20], rf_importances[rf_indices][:20], color='teal')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, 'rf_feature_importance.png'))
        
        # 2.2. Permutation Importance (more reliable)
        print("Calculating permutation importance...")
        perm_importance = permutation_importance(
            rf_selector, X_test_processed, y_test, n_repeats=5, random_state=42
        )
        perm_indices = perm_importance.importances_mean.argsort()[::-1]
        
        feature_selection_results['permutation'] = {
            'feature_names': feature_names[perm_indices],
            'importance_values': perm_importance.importances_mean[perm_indices],
            'std_devs': perm_importance.importances_std[perm_indices]
        }
        
        # Save permutation importance visualization
        plt.figure(figsize=(12, 8))
        plt.barh(feature_names[perm_indices][:20], 
                perm_importance.importances_mean[perm_indices][:20],
                xerr=perm_importance.importances_std[perm_indices][:20], 
                color='teal')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, 'permutation_importance.png'))
        
        # 2.3. Recursive Feature Elimination with Cross-Validation
        print("Running Recursive Feature Elimination...")
        min_features_to_select = 10  # Minimum number of features to consider
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            step=1,
            cv=5,
            scoring='roc_auc',
            min_features_to_select=min_features_to_select
        )
        
        # Use a smaller subset if dataset is large
        if X_train_processed.shape[0] > 10000:
            sample_idx = np.random.choice(X_train_processed.shape[0], 10000, replace=False)
            rfecv.fit(X_train_processed[sample_idx], y_train.iloc[sample_idx])
        else:
            rfecv.fit(X_train_processed, y_train)
            
        rfe_ranking = rfecv.ranking_
        rfe_mask = rfe_ranking == 1
        
        feature_selection_results['rfe'] = {
            'feature_names': feature_names[rfe_mask],
            'n_features': sum(rfe_mask)
        }
        
        # Plot number of features vs CV score
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + min_features_to_select), 
         rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of features selected")
        plt.ylabel("CV Score (ROC AUC)")
        plt.title("RFE Cross-validation Scores")
        plt.savefig(os.path.join(models_path, 'rfe_cv_scores.png'))
        
        # 2.4. Mutual Information for Classification (information gain)
        print("Calculating mutual information...")
        mutual_info = mutual_info_classif(X_train_processed, y_train, random_state=42)
        mutual_indices = np.argsort(mutual_info)[::-1]
        
        feature_selection_results['mutual_info'] = {
            'feature_names': feature_names[mutual_indices],
            'importance_values': mutual_info[mutual_indices]
        }
        
        # 3. Aggregate feature selection results across methods
        print("Aggregating feature importance results...")
        
        # Create a scoring system based on multiple methods
        feature_scores = pd.DataFrame(index=feature_names)
        
        # Random Forest scores
        rf_scores = pd.Series(
            rf_importances, 
            index=feature_names
        ).sort_values(ascending=False)
        feature_scores['rf_rank'] = rf_scores.rank(ascending=False)
        
        # Permutation scores
        perm_scores = pd.Series(
            perm_importance.importances_mean,
            index=feature_names
        ).sort_values(ascending=False)
        feature_scores['perm_rank'] = perm_scores.rank(ascending=False)
        
        # RFE scores (lower is better)
        rfe_scores = pd.Series(
            rfecv.ranking_,
            index=feature_names
        )
        feature_scores['rfe_rank'] = rfe_scores.rank()
        
        # Mutual Information scores
        mi_scores = pd.Series(
            mutual_info,
            index=feature_names
        ).sort_values(ascending=False)
        feature_scores['mi_rank'] = mi_scores.rank(ascending=False)
        
        # Calculate average rank (lower is better)
        feature_scores['avg_rank'] = feature_scores.mean(axis=1)
        
        # Select top features based on aggregate ranking
        top_k_features = 20  # Choose how many features to keep
        top_features = feature_scores.sort_values('avg_rank').index[:top_k_features].tolist()
        print(f"Top {top_k_features} features selected: {top_features}")
        
        # Save feature ranking
        feature_scores.sort_values('avg_rank').to_csv(
            os.path.join(models_path, 'feature_ranking.csv')
        ) 
                
        # 5. Define feature selection strategies for pipelines
        feature_selection_strategies = {
            'top_features': Pipeline([
                ('selector', MultiMethodFeatureSelector(feature_names, top_features))
            ])
            # PCA constantly gave lower AUC and Accuracy as comapred to above hence can exclude 
            # ,'pca': Pipeline([ 
            #     ('selector', PCASelector(n_components=0.95))
            # ]),
            # 'combined': Pipeline([
            #     ('selector1', MultiMethodFeatureSelector(feature_names, top_features)),
            #     ('selector2', PCASelector(n_components=0.95))
            # ])
        }
        
        # Train and evaluate each model with different feature selection strategies
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model_results["booking_probability"][name] = {}
            
            for fs_name, fs_pipeline in feature_selection_strategies.items():
                print(f"  Using feature selection: {fs_name}")
                
                # Create pipeline with preprocessing and feature selection
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('feature_selection', fs_pipeline),
                    ('model', model)
                ])
                
                # Hyperparameter grid for each model type
                if name == 'Logistic Regression':
                    param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]} 
                elif name == 'Random Forest':
                    param_grid = {'model__n_estimators': [50, 100, 200],
                                'model__max_depth': [10, 20, None]}
                elif name == 'Gradient Boosting':
                    param_grid = {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1, 0.2],
                        'model__max_depth': [3, 5, 7]
                    }
                elif name == 'XGBoost':
                    param_grid = {'model__n_estimators': [100, 200],
                                'model__learning_rate': [0.01, 0.1, 0.2],
                                'model__max_depth': [3, 5, 7]}
                elif name == 'Neural Network':
                    param_grid = {
                        'model__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                        'model__activation': ['relu', 'tanh'],
                        'model__alpha': [0.0001, 0.001],
                        'model__learning_rate_init': [0.001, 0.01]
                    }
    
                # Cross-validation grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=1
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                # Store results
                model_results["booking_probability"][name][fs_name] = {
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "roc_auc": float(roc_auc),
                    "best_params": grid_search.best_params_
                }
                
                print(f"    {fs_name} - AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
                
                # Save the best model
                model_path = os.path.join(models_path, f"{name.lower().replace(' ', '_')}_{fs_name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                    
                # If using PCA, save explained variance information
                if fs_name in ['pca', 'combined']:
                    if hasattr(best_model.named_steps['feature_selection'].named_steps.get('selector2', None), 'explained_variance_ratio_'):
                        pca_info = {
                            'n_components': int(best_model.named_steps['feature_selection'].named_steps['selector2'].n_components_selected_),
                            'explained_variance_ratio': best_model.named_steps['feature_selection'].named_steps['selector2'].explained_variance_ratio_.tolist(),
                            'cumulative_variance': np.cumsum(best_model.named_steps['feature_selection'].named_steps['selector2'].explained_variance_ratio_).tolist()
                        }
                        with open(os.path.join(models_path, f"{name.lower().replace(' ', '_')}_{fs_name}_pca_info.json"), 'w') as f:
                            json.dump(pca_info, f)
                            
                        # Plot PCA explained variance
                        plt.figure(figsize=(10, 6))
                        plt.plot(np.cumsum(best_model.named_steps['feature_selection'].named_steps['selector2'].explained_variance_ratio_))
                        plt.xlabel('Number of Components')
                        plt.ylabel('Cumulative Explained Variance')
                        plt.title('PCA Explained Variance')
                        plt.grid(True)
                        plt.savefig(os.path.join(models_path, f"{name.lower().replace(' ', '_')}_{fs_name}_pca_variance.png"))
        
        # Create comparative visualization for all models and feature selection strategies
        # Reshape results for plotting
        results_for_plot = []
        for model_name, fs_results in model_results["booking_probability"].items():
            for fs_name, metrics in fs_results.items():
                results_for_plot.append({
                    'Model': model_name,
                    'Feature Selection': fs_name,
                    'AUC': metrics['roc_auc'],
                    'Accuracy': metrics['accuracy'],
                    'F1': metrics['f1_score']
                })
                
        results_df = pd.DataFrame(results_for_plot)
        
        # Plot AUC comparison
        plt.figure(figsize=(15, 8))
        sns.barplot(x='Model', y='AUC', hue='Feature Selection', data=results_df)
        plt.title('Model Performance (AUC) by Feature Selection Method')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Feature Selection')
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, 'model_feature_selection_comparison.png'))
        
        # Find overall best model
        best_model_info = results_df.loc[results_df['AUC'].idxmax()]
        best_model_name = best_model_info['Model']
        best_fs_name = best_model_info['Feature Selection']
        best_model_path = os.path.join(models_path, f"{best_model_name.lower().replace(' ', '_')}_{best_fs_name}_model.pkl")
        
        print(f"\nBest model: {best_model_name} with {best_fs_name} feature selection (AUC: {best_model_info['AUC']:.4f})")
     
    return make_serializable({ 
        'processed_data_path': processed_data_path,
        'models_directory': models_path,
        'best_model_path': best_model_path,   
        'feature_names_path': feature_names_path
    })
    
def get_price_effect(result, decrease=True):
    """Extract the effect of price decrease or increase from price response data"""
    if 'price_response' not in result:
        return None
    
    try:
        price_variations, probs = zip(*result['price_response'])
        
        if decrease:
            # Find effect of 10% price decrease (0.9)
            target_var = 0.9
        else:
            # Find effect of 10% price increase (1.1)
            target_var = 1.1
        
        # Find closest variation to target
        closest_idx = min(range(len(price_variations)), key=lambda i: abs(price_variations[i] - target_var))
        closest_var = price_variations[closest_idx]
        prob_effect = probs[closest_idx] - result.get('avg_booking_prob', 0)
        
        return float(prob_effect) if isinstance(prob_effect, (np.number, float)) else prob_effect
    except Exception as e:
        print(f"Error calculating price effect: {e}")
        return 0.0

def price_elasticity_analysis(**kwargs):
    """
    Performs comprehensive price elasticity analysis to optimize pricing strategies.
    
    This function analyzes the relationship between price and booking probability,
    calculates price elasticity coefficients, and provides optimized pricing recommendations
    based on overall data and various market segments.
    """
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='train_models')
    
    # Create output directories
    REPORT_PATH = os.path.join(OUTPUT_PATH, 'elasticity_analysis')
    os.makedirs(REPORT_PATH, exist_ok=True)
    os.makedirs(os.path.join(REPORT_PATH, 'visualizations'), exist_ok=True)
    
    # Load processed data and model
    df = pd.read_csv(paths['processed_data_path']) 
    
    with open(paths['best_model_path'], 'rb') as f:
        model = pickle.load(f)

    with open(paths['feature_names_path'], 'rb') as f:
        all_features = pickle.load(f) 
 
    # Prepare data for elasticity analysis
    X = df[all_features].copy()
    y = df['booking_status']
    
    # Initialize results storage
    elasticity_results = {
        "overall_elasticity": None,
        "elasticity_by_segment": {},
        "optimum_price_points": {}
    }
    
    # Define price modification range with finer granularity
    price_modifications = np.linspace(0.5, 1.5, 21)  # 50% to 150% of original price
    
    # Define helper function for log elasticity calculation if not already defined
    def calculate_log_elasticity(price_mods, probabilities):
        """Calculate elasticity coefficient from price modifications and probabilities"""
        if len(price_mods) != len(probabilities) or len(price_mods) < 2:
            return None
        
        # Filter out zero values that would cause log calculation issues
        valid_indices = [i for i in range(len(probabilities)) if probabilities[i] > 0]
        if len(valid_indices) < 2:
            return None
        
        x = np.log([price_mods[i] for i in valid_indices])
        y = np.log([probabilities[i] for i in valid_indices])
        
        # Add constant for intercept
        X_const = sm.add_constant(x)
        model = sm.OLS(y, X_const).fit()
        
        # The coefficient of log(price) is the elasticity
        return float(model.params[1]) if len(model.params) > 1 else None
    #------------------------------------------------------------------------------
    # 1. Calculate Overall Price Elasticity and Revenue Impact
    #------------------------------------------------------------------------------
    
    # Calculate booking probability at different price points
    base_price = X['price'].mean()
    base_X = X.copy()
    base_predictions = model.predict(base_X)
    base_booking_prob = np.mean(base_predictions)
    base_expected_revenue = base_price * base_booking_prob
    
    price_elasticity_results = []
    
    for mod in price_modifications:
        X_modified = X.copy()
        X_modified['price'] = X_modified['price'] * mod
        
        # Predict booking probability
        predictions = model.predict(X_modified)
        avg_prob = np.mean(predictions)
        
        # Calculate expected revenue at this price point
        avg_price = X_modified['price'].mean()
        expected_revenue = avg_price * avg_prob
        revenue_change = (expected_revenue - base_expected_revenue) / base_expected_revenue
        
        price_elasticity_results.append({
            'price_modification': mod,
            'avg_booking_probability': avg_prob,
            'avg_price': avg_price,
            'expected_revenue': expected_revenue,
            'revenue_change': revenue_change
        })
    
    # Convert to DataFrame
    elasticity_df = pd.DataFrame(price_elasticity_results)
    
    # Calculate point elasticity (% change in demand / % change in price)
    elasticity_df['elasticity'] = np.nan
    for i in range(1, len(elasticity_df)):
        price_change = (elasticity_df.loc[i, 'price_modification'] - elasticity_df.loc[i-1, 'price_modification']) / elasticity_df.loc[i-1, 'price_modification']
        prob_change = (elasticity_df.loc[i, 'avg_booking_probability'] - elasticity_df.loc[i-1, 'avg_booking_probability']) / elasticity_df.loc[i-1, 'avg_booking_probability']
        
        if price_change != 0:
            elasticity_df.loc[i, 'elasticity'] = prob_change / price_change
    
    # Find optimal price point (maximum revenue)
    optimal_idx = elasticity_df['expected_revenue'].idxmax()
    optimal_price_mod = elasticity_df.loc[optimal_idx, 'price_modification']
    optimal_price = elasticity_df.loc[optimal_idx, 'avg_price']
    optimal_booking_prob = elasticity_df.loc[optimal_idx, 'avg_booking_probability']
    optimal_revenue = elasticity_df.loc[optimal_idx, 'expected_revenue']
    
    # Save detailed elasticity results
    elasticity_df.to_csv(os.path.join(REPORT_PATH, 'price_elasticity_detailed.csv'), index=False)
    
    # Calculate overall elasticity using log-log model
    # Create price buckets for regression analysis
    price_ranges = pd.qcut(df['price'], 10, labels=False)
    df['price_bucket'] = price_ranges
    
    # Calculate booking rate for each price bucket
    booking_rates = df.groupby('price_bucket')['booking_status'].mean()
    median_prices = df.groupby('price_bucket')['price'].median()
    
    regression_elasticity_df = pd.DataFrame({
        'median_price': median_prices,
        'booking_rate': booking_rates
    })
    
    mask = (regression_elasticity_df['booking_rate'] > 0) & (regression_elasticity_df['median_price'] > 0)
    if sum(mask) > 1:  # Need at least 2 points for regression
        X_reg = np.log(regression_elasticity_df['median_price'][mask])
        y_reg = np.log(regression_elasticity_df['booking_rate'][mask])
        
        # Add constant for intercept
        X_const = sm.add_constant(X_reg)
        reg_model = sm.OLS(y_reg, X_const).fit()
        
        # The coefficient of log(price) is the elasticity
        overall_elasticity = reg_model.params[1]
        elasticity_results["overall_elasticity"] = overall_elasticity
        
        # Save optimal pricing information
        with open(os.path.join(REPORT_PATH, 'optimal_pricing.txt'), 'w') as f:
            f.write(f"Optimal Price Modification: {optimal_price_mod:.2f} (or {(optimal_price_mod-1)*100:.1f}% {'increase' if optimal_price_mod > 1 else 'decrease'})\n")
            f.write(f"Optimal Average Price: ${optimal_price:.2f}\n")
            f.write(f"Booking Probability at Optimal Price: {optimal_booking_prob:.4f}\n")
            f.write(f"Expected Revenue at Optimal Price: ${optimal_revenue:.2f}\n")
            f.write(f"Revenue Improvement: {elasticity_df.loc[optimal_idx, 'revenue_change']*100:.2f}%\n")
            
            # Add key elasticity insights
            mid_idx = len(elasticity_df) // 2
            mid_elasticity = elasticity_df.iloc[mid_idx]['elasticity']
            if not pd.isna(mid_elasticity):
                f.write(f"\nPrice Elasticity (at current price point): {mid_elasticity:.4f}\n")
                if abs(mid_elasticity) > 1:
                    f.write("The market is highly elastic - small price changes cause large booking probability changes.\n")
                    f.write(f"A 10% price increase leads to approximately {abs(mid_elasticity*10):.1f}% decrease in bookings.\n")
                else:
                    f.write("The market is inelastic - price changes have less impact on booking probability.\n")
                    f.write(f"A 10% price increase leads to approximately {abs(mid_elasticity*10):.1f}% decrease in bookings.\n")
            
            f.write(f"\nOverall Log-Log Elasticity Model: {overall_elasticity:.4f}\n")
    
    #------------------------------------------------------------------------------
    # 2. Create Visualizations for Overall Elasticity Analysis
    #------------------------------------------------------------------------------
    
    # Plot 1: Overall price elasticity (log-log model)
    plt.figure(figsize=(10, 6))
    plt.scatter(regression_elasticity_df['median_price'], regression_elasticity_df['booking_rate'], alpha=0.7)
    
    # Plot the fitted curve from log-log model
    if 'reg_model' in locals():
        sorted_prices = np.sort(regression_elasticity_df['median_price'])
        X_pred = np.log(sorted_prices)
        X_pred_const = sm.add_constant(X_pred)
        y_pred = np.exp(reg_model.predict(X_pred_const))
        
        plt.plot(sorted_prices, y_pred, 'r-', linewidth=2)
    
    plt.title(f'Price Elasticity of Demand (Coefficient: {overall_elasticity:.2f})')
    plt.xlabel('Price ($)')
    plt.ylabel('Booking Rate')
    plt.grid(True, alpha=0.3)
    
    # Add elasticity interpretation
    if overall_elasticity < 0:
        interpretation = f"Elastic demand (ε={overall_elasticity:.2f}): 1% price increase leads to {abs(overall_elasticity):.2f}% decrease in bookings"
    else:
        interpretation = f"Inelastic demand (ε={overall_elasticity:.2f}): Price has little effect on bookings"
        
    plt.annotate(
        interpretation, 
        xy=(0.05, 0.05),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'overall_price_elasticity.png'))
    plt.close()
    
    # Plot 2: Combined Price Elasticity and Revenue Impact
    plt.figure(figsize=(14, 8))
    
    # Plot 2.1: Booking probability vs price
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(elasticity_df['price_modification'], elasticity_df['avg_booking_probability'], 'b-', linewidth=2)
    plt.axvline(x=optimal_price_mod, color='r', linestyle='--', alpha=0.7)
    plt.title('Booking Probability vs. Price Modification')
    plt.xlabel('Price Modification Factor')
    plt.ylabel('Booking Probability')
    plt.grid(True, alpha=0.3)
    
    # Plot 2.2: Expected revenue vs price
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(elasticity_df['price_modification'], elasticity_df['expected_revenue'], 'g-', linewidth=2)
    plt.axvline(x=optimal_price_mod, color='r', linestyle='--', alpha=0.7)
    plt.scatter(optimal_price_mod, optimal_revenue, color='red', s=100, zorder=5)
    plt.title('Expected Revenue vs. Price Modification')
    plt.xlabel('Price Modification Factor')
    plt.ylabel('Expected Revenue ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2.3: Price elasticity
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(elasticity_df['price_modification'][1:], elasticity_df['elasticity'][1:], 'purple', linewidth=2)
    plt.axhline(y=-1, color='grey', linestyle='--', alpha=0.7)
    plt.title('Price Elasticity')
    plt.xlabel('Price Modification Factor')
    plt.ylabel('Elasticity')
    plt.grid(True, alpha=0.3)
    
    # Plot 2.4: Revenue change percentage
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(elasticity_df['price_modification'], elasticity_df['revenue_change']*100, 'orange', linewidth=2)
    plt.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
    plt.axvline(x=optimal_price_mod, color='r', linestyle='--', alpha=0.7)
    plt.title('Revenue Change (%)')
    plt.xlabel('Price Modification Factor')
    plt.ylabel('Revenue Change %')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'price_elasticity_comprehensive.png'))
    plt.close()
    
    #------------------------------------------------------------------------------
    # 3. Segmented Analysis
    #------------------------------------------------------------------------------
    
    segmentation_results = {}
    
    # Identify potential segmentation columns
    segment_cols = []
    if 'room_type' in df.columns:
        segment_cols.append('room_type')
    if 'property_type' in df.columns:
        segment_cols.append('property_type')
    if 'is_weekend' in df.columns:
        segment_cols.append('is_weekend')
    if 'month' in df.columns:
        # Create season from month
        df['season'] = pd.cut(
            df['month'], 
            bins=[0, 3, 6, 9, 12], 
            labels=['Winter', 'Spring', 'Summer', 'Fall']
        )
        segment_cols.append('season')
    if 'accommodates' in df.columns:
        # Create capacity segments
        df['capacity_segment'] = pd.cut(
            df['accommodates'], 
            bins=[0, 2, 4, 6, 100], 
            labels=['1-2 people', '3-4 people', '5-6 people', '7+ people']
        )
        segment_cols.append('capacity_segment')
    
    #------------------------------------------------------------------------------
    # 3.1 Room Type Analysis
    #------------------------------------------------------------------------------
    
    if 'room_type' in df.columns:
        room_types = df['room_type'].unique()
        if len(room_types) > 1:
            room_type_elasticity = {}
            room_type_optimal = {}
            
            plt.figure(figsize=(12, 8))
            
            for room_type in room_types:
                room_df = df[df['room_type'] == room_type]
                if len(room_df) < 50:  # Skip if too few samples
                    continue
                    
                room_X = room_df[all_features].copy()
                
                room_probs = []
                room_revenues = []
                base_price = room_X['price'].mean()
                
                for mod in price_modifications:
                    room_X_mod = room_X.copy()
                    room_X_mod['price'] = room_X_mod['price'] * mod
                    room_prob = np.mean(model.predict(room_X_mod))
                    room_probs.append(room_prob)
                    
                    # Calculate expected revenue
                    avg_price = room_X_mod['price'].mean()
                    expected_revenue = avg_price * room_prob
                    room_revenues.append(expected_revenue)
                
                # Find optimal price for this room type
                optimal_idx = np.argmax(room_revenues)
                optimal_mod = price_modifications[optimal_idx]
                optimal_rev = room_revenues[optimal_idx]
                
                plt.plot(price_modifications, room_probs, label=f"{room_type}")
                
                # Store results
                room_type_elasticity[room_type] = room_probs
                room_type_optimal[room_type] = {
                    'optimal_price_mod': optimal_mod,
                    'optimal_revenue': optimal_rev,
                    'base_price': base_price,
                    'recommended_price': base_price * optimal_mod
                }
            
            plt.title('Booking Probability by Room Type')
            plt.xlabel('Price Modification Factor')
            plt.ylabel('Booking Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'room_type_elasticity.png'))
            plt.close()
            
            # Plot room type revenue curves
            plt.figure(figsize=(12, 8))
            for room_type in room_types:
                if room_type in room_type_optimal:
                    room_revenues = []
                    for mod in price_modifications:
                        room_price = room_type_optimal[room_type]['base_price'] * mod
                        idx = price_modifications.tolist().index(mod)
                        room_prob = room_type_elasticity[room_type][idx]
                        room_revenues.append(room_price * room_prob)
                    
                    plt.plot(price_modifications, room_revenues, label=f"{room_type}")
                    optimal_mod = room_type_optimal[room_type]['optimal_price_mod']
                    plt.axvline(x=optimal_mod, linestyle='--', alpha=0.5)
            
            plt.title('Expected Revenue by Room Type')
            plt.xlabel('Price Modification Factor')
            plt.ylabel('Expected Revenue ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'room_type_revenue.png'))
            plt.close()
            
            # Save room type optimal pricing
            with open(os.path.join(REPORT_PATH, 'room_type_pricing.txt'), 'w') as f:
                f.write("Optimal Pricing by Room Type:\n")
                f.write("===========================\n\n")
                
                for room_type, data in room_type_optimal.items():
                    f.write(f"Room Type: {room_type}\n")
                    f.write(f"Current Average Price: ${data['base_price']:.2f}\n")
                    f.write(f"Recommended Price Modifier: {data['optimal_price_mod']:.2f}x\n")
                    f.write(f"Recommended Price: ${data['recommended_price']:.2f}\n")
                    f.write(f"Expected Revenue at Recommended Price: ${data['optimal_revenue']:.2f}\n\n")
            
            segmentation_results['room_type'] = {
                'elasticity': room_type_elasticity,
                'optimal': room_type_optimal
            }
            
            # Store in overall results
            elasticity_results["elasticity_by_segment"]["room_type"] = {
                k: calculate_log_elasticity(price_modifications, v) for k, v in room_type_elasticity.items()
            }
            elasticity_results["optimum_price_points"]["room_type"] = {
                k: data['recommended_price'] for k, data in room_type_optimal.items()
            }
    
    #------------------------------------------------------------------------------
    # 3.2 Weekend/Weekday Analysis
    #------------------------------------------------------------------------------
    
    if 'is_weekend' in df.columns:
        weekend_df = df[df['is_weekend'] == 1]
        weekday_df = df[df['is_weekend'] == 0]
        
        weekend_X = weekend_df[all_features].copy()
        weekday_X = weekday_df[all_features].copy()
        
        weekend_elasticity = []
        weekday_elasticity = []
        weekend_revenue = []
        weekday_revenue = []
        
        weekend_base_price = weekend_X['price'].mean() if len(weekend_X) > 0 else 0
        weekday_base_price = weekday_X['price'].mean() if len(weekday_X) > 0 else 0
        
        for mod in price_modifications:
            # Weekend analysis
            if len(weekend_X) > 0:
                weekend_X_mod = weekend_X.copy()
                weekend_X_mod['price'] = weekend_X_mod['price'] * mod
                weekend_prob = np.mean(model.predict(weekend_X_mod))
                weekend_elasticity.append(weekend_prob)
                weekend_revenue.append(weekend_X_mod['price'].mean() * weekend_prob)
            
            # Weekday analysis
            if len(weekday_X) > 0:
                weekday_X_mod = weekday_X.copy()
                weekday_X_mod['price'] = weekday_X_mod['price'] * mod
                weekday_prob = np.mean(model.predict(weekday_X_mod))
                weekday_elasticity.append(weekday_prob)
                weekday_revenue.append(weekday_X_mod['price'].mean() * weekday_prob)
        
        # Determine optimal price mods
        weekend_optimal_idx = np.argmax(weekend_revenue) if weekend_revenue else 0
        weekday_optimal_idx = np.argmax(weekday_revenue) if weekday_revenue else 0
        
        weekend_optimal_mod = price_modifications[weekend_optimal_idx] if weekend_revenue else 1.0
        weekday_optimal_mod = price_modifications[weekday_optimal_idx] if weekday_revenue else 1.0
        
        # Enhanced visualization - booking prob and revenue side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if weekend_elasticity and weekday_elasticity:
            # Booking probability plot
            ax1.plot(price_modifications, weekend_elasticity, 'r-', label='Weekend', linewidth=2)
            ax1.plot(price_modifications, weekday_elasticity, 'b-', label='Weekday', linewidth=2)
            ax1.set_title('Booking Probability: Weekend vs Weekday')
            ax1.set_xlabel('Price Modification Factor')
            ax1.set_ylabel('Booking Probability')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Revenue plot
            ax2.plot(price_modifications, weekend_revenue, 'r-', label='Weekend', linewidth=2)
            ax2.plot(price_modifications, weekday_revenue, 'b-', label='Weekday', linewidth=2)
            ax2.axvline(x=weekend_optimal_mod, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=weekday_optimal_mod, color='b', linestyle='--', alpha=0.5)
            ax2.set_title('Expected Revenue: Weekend vs Weekday')
            ax2.set_xlabel('Price Modification Factor')
            ax2.set_ylabel('Expected Revenue ($)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'weekend_weekday_comprehensive.png'))
        plt.close()
        
        # Save weekend/weekday optimal pricing recommendations
        with open(os.path.join(REPORT_PATH, 'weekend_weekday_pricing.txt'), 'w') as f:
            f.write("Weekend vs Weekday Pricing Recommendations:\n")
            f.write("=======================================\n\n")
            
            f.write("Weekend Pricing:\n")
            f.write(f"Current Average Price: ${weekend_base_price:.2f}\n")
            f.write(f"Recommended Price Modifier: {weekend_optimal_mod:.2f}x\n")
            f.write(f"Recommended Weekend Price: ${weekend_base_price * weekend_optimal_mod:.2f}\n")
            if weekend_revenue:
                f.write(f"Expected Revenue at Recommended Price: ${weekend_revenue[weekend_optimal_idx]:.2f}\n\n")
            
            f.write("Weekday Pricing:\n")
            f.write(f"Current Average Price: ${weekday_base_price:.2f}\n")
            f.write(f"Recommended Price Modifier: {weekday_optimal_mod:.2f}x\n")
            f.write(f"Recommended Weekday Price: ${weekday_base_price * weekday_optimal_mod:.2f}\n")
            if weekday_revenue:
                f.write(f"Expected Revenue at Recommended Price: ${weekday_revenue[weekday_optimal_idx]:.2f}\n\n")
            
            # Add differential pricing strategy
            if weekend_optimal_mod > weekday_optimal_mod:
                premium = ((weekend_base_price * weekend_optimal_mod) / (weekday_base_price * weekday_optimal_mod) - 1) * 100
                f.write(f"Recommendation: Set weekend prices {premium:.1f}% higher than weekday prices\n")
            elif weekday_optimal_mod > weekend_optimal_mod:
                premium = ((weekday_base_price * weekday_optimal_mod) / (weekend_base_price * weekend_optimal_mod) - 1) * 100
                f.write(f"Recommendation: Set weekday prices {premium:.1f}% higher than weekend prices\n")
            else:
                f.write("Recommendation: Similar pricing for weekends and weekdays is optimal\n")
        
        segmentation_results['day_type'] = {
            'weekend': {
                'elasticity': weekend_elasticity,
                'revenue': weekend_revenue,
                'optimal_mod': weekend_optimal_mod
            },
            'weekday': {
                'elasticity': weekday_elasticity,
                'revenue': weekday_revenue,
                'optimal_mod': weekday_optimal_mod
            }
        }
        
        # Store in overall results
        elasticity_results["elasticity_by_segment"]["day_type"] = {
            "Weekend": calculate_log_elasticity(price_modifications, weekend_elasticity) if weekend_elasticity else None,
            "Weekday": calculate_log_elasticity(price_modifications, weekday_elasticity) if weekday_elasticity else None
        }
        elasticity_results["optimum_price_points"]["day_type"] = {
            "Weekend": weekend_base_price * weekend_optimal_mod,
            "Weekday": weekday_base_price * weekday_optimal_mod
        }
    
    #------------------------------------------------------------------------------
    # 3.3 Seasonal Analysis
    #------------------------------------------------------------------------------
    
    if 'season' in df.columns and len(df['season'].unique()) > 1:
        season_elasticity = {}
        season_revenue = {}
        season_optimal = {}
        season_base_price = {}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            if len(season_df) < 30:  # Skip if too few samples
                continue
                
            season_X = season_df[all_features].copy()
            season_base_price[season] = season_X['price'].mean()
            
            season_probs = []
            season_revs = []
            
            for mod in price_modifications:
                season_X_mod = season_X.copy()
                season_X_mod['price'] = season_X_mod['price'] * mod
                season_prob = np.mean(model.predict(season_X_mod))
                season_probs.append(season_prob)
                
                # Calculate expected revenue
                avg_price = season_X_mod['price'].mean()
                expected_revenue = avg_price * season_prob
                season_revs.append(expected_revenue)
            
            # Find optimal price for this season
            optimal_idx = np.argmax(season_revs)
            optimal_mod = price_modifications[optimal_idx]
            optimal_rev = season_revs[optimal_idx]
            
            ax1.plot(price_modifications, season_probs, label=season)
            ax2.plot(price_modifications, season_revs, label=season)
            ax2.axvline(x=optimal_mod, linestyle='--', alpha=0.5, color=ax2.lines[-1].get_color())
            
            # Store results
            season_elasticity[season] = season_probs
            season_revenue[season] = season_revs
            season_optimal[season] = {
                'optimal_mod': optimal_mod,
                'optimal_revenue': optimal_rev
            }
        
        ax1.set_title('Booking Probability by Season')
        ax1.set_xlabel('Price Modification Factor')
        ax1.set_ylabel('Booking Probability')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Expected Revenue by Season')
        ax2.set_xlabel('Price Modification Factor')
        ax2.set_ylabel('Expected Revenue ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'seasonal_comprehensive.png'))
        plt.close()
        
        # Save seasonal pricing recommendations
        with open(os.path.join(REPORT_PATH, 'seasonal_pricing.txt'), 'w') as f:
            f.write("Seasonal Pricing Recommendations:\n")
            f.write("==============================\n\n")
            
            for season, data in season_optimal.items():
                base_price = season_base_price.get(season, 0)
                f.write(f"Season: {season}\n")
                f.write(f"Current Average Price: ${base_price:.2f}\n")
                f.write(f"Recommended Price Modifier: {data['optimal_mod']:.2f}x\n")
                f.write(f"Recommended Price: ${base_price * data['optimal_mod']:.2f}\n")
                f.write(f"Expected Revenue at Recommended Price: ${data['optimal_revenue']:.2f}\n\n")
            
            # Calculate and recommend seasonal pricing strategy
            if len(season_optimal) > 1:
                seasons = list(season_optimal.keys())
                f.write("Seasonal Price Differential Strategy:\n")
                f.write("--------------------------------\n")
                
                # Find highest and lowest seasons by optimal revenue
                seasons_by_revenue = sorted(seasons, key=lambda x: season_optimal[x]['optimal_revenue'], reverse=True)
                high_season = seasons_by_revenue[0]
                low_season = seasons_by_revenue[-1]
                
                high_price = season_base_price[high_season] * season_optimal[high_season]['optimal_mod']
                low_price = season_base_price[low_season] * season_optimal[low_season]['optimal_mod']
                
                price_diff = (high_price / low_price - 1) * 100 if low_price > 0 else 0
                
                f.write(f"High-revenue season ({high_season}): ${high_price:.2f}\n")
                f.write(f"Low-revenue season ({low_season}): ${low_price:.2f}\n")
                f.write(f"Recommended price differential: {price_diff:.1f}%\n")
        
        segmentation_results['seasonal'] = {
            'elasticity': season_elasticity,
            'revenue': season_revenue,
            'optimal': season_optimal,
            'base_price': season_base_price
        }
        
        # Store in overall results
        elasticity_results["elasticity_by_segment"]["season"] = {
            k: calculate_log_elasticity(price_modifications, v) for k, v in season_elasticity.items()
        }
        elasticity_results["optimum_price_points"]["season"] = {
            k: season_base_price[k] * season_optimal[k]['optimal_mod'] for k in season_elasticity.keys()
        }
    
    #------------------------------------------------------------------------------
    # 3.4 Capacity Segment Analysis
    #------------------------------------------------------------------------------
    
    if 'capacity_segment' in df.columns and len(df['capacity_segment'].unique()) > 1:
        capacity_elasticity = {}
        capacity_revenue = {}
        capacity_optimal = {}
        capacity_base_price = {}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for capacity in df['capacity_segment'].unique():
            capacity_df = df[df['capacity_segment'] == capacity]
            if len(capacity_df) < 30:  # Skip if too few samples
                continue
                
            capacity_X = capacity_df[all_features].copy()
            capacity_base_price[capacity] = capacity_X['price'].mean()
            
            capacity_probs = []
            capacity_revs = []
            
            for mod in price_modifications:
                capacity_X_mod = capacity_X.copy()
                capacity_X_mod['price'] = capacity_X_mod['price'] * mod
                capacity_prob = np.mean(model.predict(capacity_X_mod))
                capacity_probs.append(capacity_prob)
                
                # Calculate expected revenue
                avg_price = capacity_X_mod['price'].mean()
                expected_revenue = avg_price * capacity_prob
                capacity_revs.append(expected_revenue)
            
            # Find optimal price for this capacity segment
            optimal_idx = np.argmax(capacity_revs)
            optimal_mod = price_modifications[optimal_idx]
            optimal_rev = capacity_revs[optimal_idx]
            
            ax1.plot(price_modifications, capacity_probs, label=capacity)
            ax2.plot(price_modifications, capacity_revs, label=capacity)
            ax2.axvline(x=optimal_mod, linestyle='--', alpha=0.5, color=ax2.lines[-1].get_color())
            
            # Store results
            capacity_elasticity[capacity] = capacity_probs
            capacity_revenue[capacity] = capacity_revs
            capacity_optimal[capacity] = {
                'optimal_mod': optimal_mod,
                'optimal_revenue': optimal_rev
            }
        
        ax1.set_title('Booking Probability by Capacity')
        ax1.set_xlabel('Price Modification Factor')
        ax1.set_ylabel('Booking Probability')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Expected Revenue by Capacity')
        ax2.set_xlabel('Price Modification Factor')
        ax2.set_ylabel('Expected Revenue ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'capacity_comprehensive.png'))
        plt.close()
        
        # Save capacity pricing recommendations
        with open(os.path.join(REPORT_PATH, 'capacity_pricing.txt'), 'w') as f:
            f.write("Capacity-based Pricing Recommendations:\n")
            f.write("===================================\n\n")
            
            for capacity, data in capacity_optimal.items():
                base_price = capacity_base_price.get(capacity, 0)
                f.write(f"Capacity: {capacity}\n")
                f.write(f"Current Average Price: ${base_price:.2f}\n")
                f.write(f"Recommended Price Modifier: {data['optimal_mod']:.2f}x\n")
                f.write(f"Recommended Price: ${base_price * data['optimal_mod']:.2f}\n")
                f.write(f"Expected Revenue at Recommended Price: ${data['optimal_revenue']:.2f}\n\n")
            
            # Calculate and recommend capacity pricing strategy
            if len(capacity_optimal) > 1:
                capacities = list(capacity_optimal.keys())
                f.write("Capacity-based Price Differential Strategy:\n")
                f.write("-------------------------------------\n")
                
                # Sort capacities for logical comparison (smallest to largest)
                capacities_sorted = sorted(capacities, key=lambda x: ['1-2 people', '3-4 people', '5-6 people', '7+ people'].index(x) if x in ['1-2 people', '3-4 people', '5-6 people', '7+ people'] else 999)
                
                # Compare smallest and largest capacity
                smallest_capacity = capacities_sorted[0]
                largest_capacity = capacities_sorted[-1]
                
                small_price = capacity_base_price[smallest_capacity] * capacity_optimal[smallest_capacity]['optimal_mod']
                large_price = capacity_base_price[largest_capacity] * capacity_optimal[largest_capacity]['optimal_mod']
                
                price_diff = (large_price / small_price - 1) * 100 if small_price > 0 else 0
                
                f.write(f"Smallest capacity ({smallest_capacity}): ${small_price:.2f}\n")
                f.write(f"Largest capacity ({largest_capacity}): ${large_price:.2f}\n")
                f.write(f"Recommended price differential: {price_diff:.1f}%\n")
                f.write(f"Recommendation: Price per additional guest: ${(large_price - small_price) / (capacities_sorted.index(largest_capacity) - capacities_sorted.index(smallest_capacity)):.2f}\n")
        
        segmentation_results['capacity'] = {
            'elasticity': capacity_elasticity,
            'revenue': capacity_revenue,
            'optimal': capacity_optimal,
            'base_price': capacity_base_price
        }
        
        # Store in overall results
        elasticity_results["elasticity_by_segment"]["capacity"] = {
            k: calculate_log_elasticity(price_modifications, v) for k, v in capacity_elasticity.items()
        }
        elasticity_results["optimum_price_points"]["capacity"] = {
            k: capacity_base_price[k] * capacity_optimal[k]['optimal_mod'] for k in capacity_elasticity.keys()
        }
    
    #------------------------------------------------------------------------------
    # 3.5 Geographic Clustering Analysis
    #------------------------------------------------------------------------------
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Use geographic coordinates to cluster listings
        geo_data = df[['latitude', 'longitude']].copy()
        
        # Skip if too few samples or missing data
        if len(geo_data) > 100 and geo_data['latitude'].notna().all() and geo_data['longitude'].notna().all():
            try:  
                # Determine optimal number of clusters (2-6)
                inertia = []
                k_range = range(2, min(7, len(geo_data) // 50 + 1))  # Adjust based on data size
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(geo_data)
                    inertia.append(kmeans.inertia_)
                
                # Choose k using elbow method (simplified)
                k = k_range[min(2, len(inertia)-1)] if inertia else 3
                
                # Create clusters
                kmeans = KMeans(n_clusters=k, random_state=42)
                df['geo_cluster'] = kmeans.fit_predict(geo_data)
                
                # Calculate cluster centers for mapping
                cluster_centers = kmeans.cluster_centers_
                
                # Analyze price elasticity by geographic cluster
                geo_elasticity = {}
                geo_revenue = {}
                geo_optimal = {}
                geo_base_price = {}
                
                # Plot cluster map
                plt.figure(figsize=(12, 8))
                
                # Show listings colored by cluster
                for cluster_id in range(k):
                    cluster_data = df[df['geo_cluster'] == cluster_id]
                    plt.scatter(cluster_data['longitude'], cluster_data['latitude'], 
                                alpha=0.5, label=f'Cluster {cluster_id+1}')
                
                # Show cluster centers
                plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], 
                            s=200, marker='X', c='black', label='Cluster Centers')
                
                plt.title('Geographic Clusters of Listings')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'geographic_clusters.png'))
                plt.close()
                
                # Create subplots for pricing analysis by geographic cluster
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                for cluster_id in range(k):
                    cluster_data = df[df['geo_cluster'] == cluster_id]
                    cluster_X = cluster_data[all_features].copy()
                    
                    if len(cluster_X) < 20:  # Skip if too few samples
                        continue
                    
                    geo_base_price[f'Cluster {cluster_id+1}'] = cluster_X['price'].mean()
                    
                    cluster_probs = []
                    cluster_revs = []
                    
                    for mod in price_modifications:
                        cluster_X_mod = cluster_X.copy()
                        cluster_X_mod['price'] = cluster_X_mod['price'] * mod
                        cluster_prob = np.mean(model.predict(cluster_X_mod))
                        cluster_probs.append(cluster_prob)
                        
                        # Calculate expected revenue
                        avg_price = cluster_X_mod['price'].mean()
                        expected_revenue = avg_price * cluster_prob
                        cluster_revs.append(expected_revenue)
                    
                    # Find optimal price for this cluster
                    optimal_idx = np.argmax(cluster_revs)
                    optimal_mod = price_modifications[optimal_idx]
                    optimal_rev = cluster_revs[optimal_idx]
                    
                    ax1.plot(price_modifications, cluster_probs, label=f'Cluster {cluster_id+1}')
                    ax2.plot(price_modifications, cluster_revs, label=f'Cluster {cluster_id+1}')
                    ax2.axvline(x=optimal_mod, linestyle='--', alpha=0.5, color=ax2.lines[-1].get_color())
                    
                    # Store results
                    geo_elasticity[f'Cluster {cluster_id+1}'] = cluster_probs
                    geo_revenue[f'Cluster {cluster_id+1}'] = cluster_revs
                    geo_optimal[f'Cluster {cluster_id+1}'] = {
                        'optimal_mod': optimal_mod,
                        'optimal_revenue': optimal_rev,
                        'center_lat': cluster_centers[cluster_id][0],
                        'center_lon': cluster_centers[cluster_id][1]
                    }
                
                ax1.set_title('Booking Probability by Geographic Cluster')
                ax1.set_xlabel('Price Modification Factor')
                ax1.set_ylabel('Booking Probability')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                ax2.set_title('Expected Revenue by Geographic Cluster')
                ax2.set_xlabel('Price Modification Factor')
                ax2.set_ylabel('Expected Revenue ($)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(REPORT_PATH, 'visualizations', 'geographic_pricing.png'))
                plt.close()
                
                # Save geographic pricing recommendations
                with open(os.path.join(REPORT_PATH, 'geographic_pricing.txt'), 'w') as f:
                    f.write("Geographic Pricing Recommendations:\n")
                    f.write("=================================\n\n")
                    
                    for cluster, data in geo_optimal.items():
                        base_price = geo_base_price.get(cluster, 0)
                        f.write(f"Location: {cluster} (Center: {data['center_lat']:.4f}, {data['center_lon']:.4f})\n")
                        f.write(f"Current Average Price: ${base_price:.2f}\n")
                        f.write(f"Recommended Price Modifier: {data['optimal_mod']:.2f}x\n")
                        f.write(f"Recommended Price: ${base_price * data['optimal_mod']:.2f}\n")
                        f.write(f"Expected Revenue at Recommended Price: ${data['optimal_revenue']:.2f}\n\n")
                    
                    # If we have multiple clusters, compare them
                    if len(geo_optimal) > 1:
                        clusters = list(geo_optimal.keys())
                        f.write("Location-based Price Differential Strategy:\n")
                        f.write("-------------------------------------\n")
                        
                        # Find the cluster with the highest recommended price
                        premium_cluster = max(clusters, key=lambda x: geo_base_price.get(x, 0) * geo_optimal[x]['optimal_mod'])
                        budget_cluster = min(clusters, key=lambda x: geo_base_price.get(x, 0) * geo_optimal[x]['optimal_mod'])
                        
                        premium_price = geo_base_price[premium_cluster] * geo_optimal[premium_cluster]['optimal_mod']
                        budget_price = geo_base_price[budget_cluster] * geo_optimal[budget_cluster]['optimal_mod']
                        
                        price_diff = (premium_price / budget_price - 1) * 100 if budget_price > 0 else 0
                        
                        f.write(f"Premium location ({premium_cluster}): ${premium_price:.2f}\n")
                        f.write(f"Budget location ({budget_cluster}): ${budget_price:.2f}\n")
                        f.write(f"Recommended price differential: {price_diff:.1f}%\n")
                
                segmentation_results['geographic'] = {
                    'elasticity': geo_elasticity,
                    'revenue': geo_revenue,
                    'optimal': geo_optimal,
                    'base_price': geo_base_price,
                    'cluster_centers': cluster_centers.tolist() if 'cluster_centers' in locals() else None
                }
                
                # Store in overall results
                elasticity_results["elasticity_by_segment"]["geo_cluster"] = {
                    k: calculate_log_elasticity(price_modifications, v) for k, v in geo_elasticity.items()
                }
                elasticity_results["optimum_price_points"]["geo_cluster"] = {
                    k: geo_base_price[k] * geo_optimal[k]['optimal_mod'] for k in geo_elasticity.keys()
                }
            except Exception as e:
                print(f"Geographic clustering failed: {str(e)}")
    
    #------------------------------------------------------------------------------
    # 4. Create comprehensive pricing strategy report
    #------------------------------------------------------------------------------
    # Create comprehensive pricing strategy report
    with open(os.path.join(REPORT_PATH, 'pricing_strategy_report.txt'), 'w') as f:
        f.write("COMPREHENSIVE PRICING STRATEGY REPORT\n")
        f.write("==================================\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("----------------\n")
        f.write(f"Overall Recommended Price Adjustment: {(optimal_price_mod-1)*100:.1f}% {'increase' if optimal_price_mod > 1 else 'decrease'}\n")
        f.write(f"Expected Revenue Improvement: {elasticity_df.loc[optimal_idx, 'revenue_change']*100:.2f}%\n")
        f.write(f"Overall Price Elasticity: {elasticity_results['overall_elasticity']:.4f}\n\n")
        
        f.write("MARKET ELASTICITY INSIGHTS\n")
        f.write("-----------------------\n")
        if elasticity_results['overall_elasticity'] < -1:
            f.write("The market is highly elastic - small price changes cause large booking probability changes.\n")
            f.write(f"A 10% price increase leads to approximately {abs(elasticity_results['overall_elasticity']*10):.1f}% decrease in bookings.\n")
        else:
            f.write("The market is relatively inelastic - price changes have less impact on booking probability.\n")
            f.write(f"A 10% price increase leads to approximately {abs(elasticity_results['overall_elasticity']*10):.1f}% decrease in bookings.\n")
        f.write("\n")
        
        # Segment-specific recommendations
        f.write("SEGMENT-SPECIFIC PRICING STRATEGY\n")
        f.write("-------------------------------\n")
        
        # Room type recommendations if available
        if 'room_type' in elasticity_results["optimum_price_points"]:
            f.write("\nBy Room Type:\n")
            for room_type, price in elasticity_results["optimum_price_points"]["room_type"].items():
                f.write(f"- {room_type}: ${price:.2f}\n")
        
        # Weekend/Weekday recommendations if available
        if 'day_type' in elasticity_results["optimum_price_points"]:
            f.write("\nBy Day Type:\n")
            if 'Weekend' in elasticity_results["optimum_price_points"]["day_type"] and 'Weekday' in elasticity_results["optimum_price_points"]["day_type"]:
                weekend = elasticity_results["optimum_price_points"]["day_type"]["Weekend"]
                weekday = elasticity_results["optimum_price_points"]["day_type"]["Weekday"]
                diff = (weekend / weekday - 1) * 100 if weekday > 0 else 0
                
                f.write(f"- Weekend: ${weekend:.2f}\n")
                f.write(f"- Weekday: ${weekday:.2f}\n")
                if diff > 5:
                    f.write(f"  Recommendation: Set weekend prices {diff:.1f}% higher than weekday prices\n")
                elif diff < -5:
                    f.write(f"  Recommendation: Set weekend prices {abs(diff):.1f}% lower than weekday prices\n")
                else:
                    f.write("  Recommendation: Similar pricing for weekends and weekdays is optimal\n")
        
        # Seasonal recommendations if available
        if 'season' in elasticity_results["optimum_price_points"]:
            f.write("\nBy Season:\n")
            seasons = list(elasticity_results["optimum_price_points"]["season"].keys())
            for season in seasons:
                price = elasticity_results["optimum_price_points"]["season"][season]
                f.write(f"- {season}: ${price:.2f}\n")
            
            # Find highest and lowest seasons
            high_season = max(seasons, key=lambda x: elasticity_results["optimum_price_points"]["season"][x])
            low_season = min(seasons, key=lambda x: elasticity_results["optimum_price_points"]["season"][x])
            high_price = elasticity_results["optimum_price_points"]["season"][high_season]
            low_price = elasticity_results["optimum_price_points"]["season"][low_season]
            
            if high_price > low_price * 1.1:  # At least 10% difference
                f.write(f"  Recommendation: Implement seasonal pricing with premium prices during {high_season}\n")
                f.write(f"  and discounted rates during {low_season} ({(high_price/low_price-1)*100:.1f}% differential)\n")
        
        # Geographic recommendations if available
        if 'geo_cluster' in elasticity_results["optimum_price_points"]:
            f.write("\nBy Geographic Location:\n")
            for cluster, price in elasticity_results["optimum_price_points"]["geo_cluster"].items():
                f.write(f"- {cluster}: ${price:.2f}\n")
        
        f.write("\n\nDYNAMIC PRICING STRATEGY\n")
        f.write("----------------------\n")
        f.write("Based on the elasticity analysis, we recommend a multi-factor pricing strategy that considers:\n")
        if 'room_type' in elasticity_results["optimum_price_points"]:
            f.write("1. Room type - Different pricing for each room category\n")
        if 'day_type' in elasticity_results["optimum_price_points"]:
            f.write("2. Day of week - Weekend vs. weekday differential pricing\n")
        if 'season' in elasticity_results["optimum_price_points"]:
            f.write("3. Seasonality - Higher prices during peak seasons\n")
        if 'geo_cluster' in elasticity_results["optimum_price_points"]:
            f.write("4. Location - Premium pricing for high-demand areas\n")
        f.write("\nImplement these pricing recommendations to maximize revenue while maintaining competitive occupancy rates.\n")
    
    # Save all segmentation results
    with open(os.path.join(REPORT_PATH, 'segmentation_analysis.pkl'), 'wb') as f:
        pickle.dump(segmentation_results, f)
    
    # Save elasticity results to JSON
    with open(os.path.join(REPORT_PATH, 'elasticity_results.json'), 'w') as f:
        json.dump(elasticity_results, f, indent=4, default=json_serializer)
    
    return {
        'elasticity_analysis_completed': True,
        'optimal_price_modification': optimal_price_mod if 'optimal_price_mod' in locals() else None,
        'revenue_improvement': elasticity_df.loc[optimal_idx, 'revenue_change'] if 'optimal_idx' in locals() else None,
        'report_path': REPORT_PATH
    } 
                           
def generate_report(**kwargs):
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='price_elasticity_analysis')
    
    # Get current date for report generation
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Create a comprehensive report
    report_content = """# Airbnb Booking Probability and Price Elasticity Analysis

    ## Executive Summary
    This report presents the findings of our analysis on Airbnb booking patterns and price elasticity. 
    The analysis focuses on understanding how different factors affect booking probability and pricing optimization strategies.

    ## Data Overview
    We analyzed Airbnb listings data including property characteristics, pricing, and availability information.
    """
    
    # Create directory if it doesn't exist
    os.makedirs(REPORT_PATH, exist_ok=True)
    
    # Write the report to a file
    with open(os.path.join(REPORT_PATH, 'final_report.md'), 'w') as f:
        f.write(report_content)
    
    # Create a detailed HTML version with visualizations
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Airbnb Price Elasticity & Booking Analysis</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            root {{ 
            --primary: #FF5A5F;
            --primary-light: #FF7E82;
            --primary-dark: #E04E53;
            --secondary: #00A699;
            --dark: #2B303A;
            --light: #F7F7F7;
            --gray: #767676;
            --light-gray: #EBEBEB;
            --accent: #FFB400;
            }}

            body {{
                font-family: 'Inter', sans-serif;
                background-color: #FAFAFA;
                margin: 0;
                padding: 0;
                color: var(--dark);
                line-height: 1.6;
            }}

            header {{
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: black;
                padding: 60px 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                position: relative;
                overflow: hidden;
            }}

            header::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxkZWZzPjxwYXR0ZXJuIGlkPSJwYXR0ZXJuIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiIHBhdHRlcm5UcmFuc2Zvcm09InJvdGF0ZSg0NSkiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjA1KSIvPjwvcGF0dGVybj48L2RlZnM+PHJlY3QgZmlsbD0idXJsKCNwYXR0ZXJuKSIgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIvPjwvc3ZnPg==');
                opacity: 0.4;
            }}

            header h1 {{
                margin: 0;
                font-size: 2.8rem;
                font-family: 'Playfair Display', serif;
                font-weight: 600;
                letter-spacing: -0.5px;
                position: relative;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            header p {{
                margin-top: 15px;
                font-size: 1.2rem;
                opacity: 0.9;
                font-weight: 300;
                position: relative;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
            }}

            .container {{ 
                max-width: 1200px;
                margin: 50px auto;
                padding: 0 20px;
            }}

            .section {{
                background: white;
                padding: 40px;
                border-radius: 16px;
                margin-bottom: 50px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.05);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid rgba(0,0,0,0.03);
            }}

            .section:hover {{
                box-shadow: 0 12px 40px rgba(0,0,0,0.08);
                transform: translateY(-2px);
            }}

            h2 {{
                border-bottom: 2px solid var(--light-gray);
                padding-bottom: 15px;
                margin-bottom: 30px;
                color: var(--primary);
                font-size: 2rem;
                font-family: 'Playfair Display', serif;
                font-weight: 600;
                letter-spacing: -0.5px;
            }}

            .viz-section {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
                gap: 30px;
                margin: 40px 0;
            }}

            .viz-card {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                display: flex;
                flex-direction: column;
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                border: 1px solid rgba(0,0,0,0.03);
                background: linear-gradient(to bottom, white 0%, #fafafa 100%);
            }}

            .viz-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.05);
            }}

            .viz-card h3 {{
                text-align: center;
                margin: 0 0 20px 0;
                color: var(--dark);
                font-size: 1.4rem;
                font-weight: 600;
                padding-bottom: 15px;
                border-bottom: 1px solid var(--light-gray);
                font-family: 'Playfair Display', serif;
            }}

            .viz-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                object-fit: contain;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                transition: transform 0.3s ease;
                border: 1px solid rgba(0,0,0,0.05);
            }}

            .viz-card:hover img {{
                transform: scale(1.02);
            }}

            .viz-card p {{
                margin-top: 20px;
                font-size: 1rem;
                color: var(--gray);
                line-height: 1.6;
            }}
    
            footer {{
                text-align: center;
                padding: 40px 25px;
                font-size: 1rem;
                color: var(--gray);
                border-top: 1px solid var(--light-gray);
                margin-top: 80px;
                background: white;
            }}

            .toc {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                margin-bottom: 40px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.05);
                background: linear-gradient(to bottom, white 0%, #fafafa 100%);
                border: 1px solid rgba(0,0,0,0.03);
            }}

            .toc h3 {{
                margin-top: 0;
                font-family: 'Playfair Display', serif;
                color: var(--primary);
                font-size: 1.6rem;
                margin-bottom: 20px;
            }}

            .toc ul {{
                list-style-type: none;
                padding-left: 0;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
            }}

            .toc li {{
                margin-bottom: 12px;
                position: relative;
                padding-left: 20px;
            }}

            .toc li::before {{
                content: "→";
                position: absolute;
                left: 0;
                color: var(--primary);
            }}

            .toc a {{
                color: var(--dark);
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s ease;
                display: block;
                padding: 8px 0;
            }}

            .toc a:hover {{
                color: var(--primary);
                text-decoration: none;
            }}

            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                margin: 30px 0;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}

            th, td {{
                text-align: left;
                padding: 15px;
                border-bottom: 1px solid var(--light-gray);
            }}

            th {{
                background-color: var(--primary);
                color: white;
                font-weight: 500;
                text-transform: uppercase;
                font-size: 0.85rem;
                letter-spacing: 0.5px;
            }}

            tr:nth-child(even) {{
                background-color: rgba(0,0,0,0.01);
            }}

            tr:hover {{
                background-color: rgba(0,166,153,0.05);
            }}
 
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .section {{
                animation: fadeInUp 0.6s ease forwards;
            }}

            .section:nth-child(1) {{ animation-delay: 0.1s; }}
            .section:nth-child(2) {{ animation-delay: 0.2s; }}
            .section:nth-child(3) {{ animation-delay: 0.3s; }}
            .section:nth-child(4) {{ animation-delay: 0.4s; }}
            .section:nth-child(5) {{ animation-delay: 0.5s; }}
            .section:nth-child(6) {{ animation-delay: 0.6s; }}
            .section:nth-child(7) {{ animation-delay: 0.7s; }}
 
            @media (max-width: 768px) {{
                header h1 {{
                    font-size: 2.2rem;
                }}
                
                .viz-section {{
                    grid-template-columns: 1fr;
                }}
                
                .toc ul {{
                    grid-template-columns: 1fr;
                }}
                
                .section {{
                    padding: 30px 20px;
                }}
            }}
 
            .decorative-circle {{
                position: absolute;
                border-radius: 50%;
                opacity: 0.1;
                z-index: 0;
            }}
            
            .circle-1 {{
                width: 200px;
                height: 200px;
                background: var(--primary);
                top: -50px;
                right: -50px;
            }}
            
            .circle-2 {{
                width: 150px;
                height: 150px;
                background: var(--secondary);
                bottom: -30px;
                left: -30px;
            }}
 
            .viz-card {{
                position: relative;
            }}
            
            .viz-card:hover::after {{ 
                position: absolute;
                bottom: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 0.8rem;
                opacity: 0;
                animation: fadeIn 0.3s ease forwards;
            }}
            
            @keyframes fadeIn {{
                to {{ opacity: 1; }}
            }}
        </style>
    </head>
    <body>

    <header>
        <div class="decorative-circle circle-1"></div>
        <div class="decorative-circle circle-2"></div>
        <h1>Airbnb Booking Probability & Price Elasticity Report</h1>
        <p>Generated on: <strong>April 19, 2025</strong></p>
    </header>

    <div class="container">
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#price-elasticity">Price Elasticity Analysis</a></li>
                <li><a href="#geographical">Geographical Analysis</a></li>
                <li><a href="#room-type">Room Type Analysis</a></li>
                <li><a href="#seasonal">Seasonal Trends</a></li>
                <li><a href="#weekday-weekend">Weekday vs Weekend</a></li>
                <li><a href="#model-performance">Model Performance</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
            </ul>
        </div>

        <div class="section" id="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report explores how Airbnb listing prices influence booking probability, uncovering key patterns across room types, dates, and other features. The visualizations below highlight elasticity effects, seasonal trends, and feature importance derived from predictive modeling.</p>
            
            <p>Our analysis reveals that price elasticity is not uniform across all listing types and times of year. By understanding these patterns, hosts can optimize their pricing strategies to maximize both occupancy and revenue.</p>
        </div>

        <div class="section" id="price-elasticity">
            <h2>Price Elasticity Analysis</h2>
            <p>Price elasticity measures how booking probability changes as price changes. Understanding this relationship is crucial for optimizing revenue.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Price Distribution</h3>
                    <img src="../visualizations/price_distribution_analysis.png" alt="Price Distribution">
                    <p>The distribution of prices across all listings, showing the range and concentration of listing prices.</p>
                </div>

                <div class="viz-card">
                    <h3>Price Elasticity</h3>
                    <img src="../elasticity_analysis/visualizations/overall_price_elasticity.png" alt="Price Elasticity">
                    <p>The relationship between price changes and booking probability, showing the overall elasticity curve.</p>
                </div>

                <div class="viz-card">
                    <h3>Comprehensive Price Elasticity</h3>
                    <img src="../elasticity_analysis/visualizations/price_elasticity_comprehensive.png" alt="Comprehensive Price Elasticity">
                    <p>A more detailed view of price elasticity accounting for different property characteristics.</p>
                </div>

                <div class="viz-card">
                    <h3>Price vs Accommodates</h3>
                    <img src="../visualizations/property_characteristics.png" alt="Price vs Accommodates">
                    <p>How listing prices vary based on the number of guests that can be accommodated.</p>
                </div>
            </div>
        </div>

        <div class="section" id="geographical">
            <h2>Geographical Analysis</h2>
            <p>Location plays a critical role in determining both optimal pricing and booking probability.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Geographic Distribution</h3>
                    <img src="../visualizations/geographic_analysis.png" alt="Geographic Distribution">
                    <p>The spatial distribution of listings, highlighting concentration in different areas.</p>
                </div>
                
                <div class="viz-card">
                    <h3>Geographic Pricing Clusters</h3>
                    <img src="../elasticity_analysis/visualizations/geographic_clusters.png" alt="Geographic Pricing Clusters">
                    <p>Analysis of how different geographical areas form distinct pricing clusters.</p>
                </div>

                <div class="viz-card">
                    <h3>Geographic Pricing </h3>
                    <img src="../elasticity_analysis/visualizations/geographic_pricing.png" alt="Geographic Pricing">
                    <p>Analysis of booking propability and revenue difference between distinct geographical clusters</p>
                </div> 
                
                <div class="viz-card">
                    <h3>Neighborhood Analysis</h3>
                    <img src="../visualizations/neighborhood_analysis.png" alt="Neighborhood Analysis">
                    <p>Detailed breakdown of how neighborhoods affect pricing and booking probability.</p>
                </div>
            </div>
        </div>

        <div class="section" id="room-type">
            <h2>Room Type Analysis</h2>
            <p>Different room types (entire homes, private rooms, shared rooms) show distinct pricing patterns and elasticities.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Room Type Analysis</h3>
                    <img src="../elasticity_analysis/visualizations/room_type_revenue.png" alt="Room Type Analysis">
                    <p>Comparison of booking patterns and prices across different room types.</p>
                </div>

                <div class="viz-card">
                    <h3>Room Type Elasticity</h3>
                    <img src="../elasticity_analysis/visualizations/room_type_elasticity.png" alt="Room Type Elasticity">
                    <p>How price sensitivity varies by room type, showing which categories are most elastic.</p>
                </div>
            </div>
        </div>

        <div class="section" id="seasonal">
            <h2>Seasonal Trends</h2>
            <p>Booking patterns and price sensitivity show strong seasonal variations.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Monthly Availability</h3>
                    <img src="../visualizations/calendar_analysis.png" alt="Monthly Availability">
                    <p>How listing availability changes throughout the year, indicating seasonal demand patterns.</p>
                </div>

                <div class="viz-card">
                    <h3>Seasonal Trends</h3>
                    <img src="../elasticity_analysis/visualizations/seasonal_comprehensive.png" alt="Seasonal Trends Comprehensive">
                    <p>Comprehensive analysis of seasonal booking patterns and price effects.</p>
                </div>

                <div class="viz-card">
                    <h3>Capacity Optimization</h3>
                    <img src="../elasticity_analysis/visualizations/capacity_comprehensive.png" alt="Capacity Optimization">
                    <p>Analysis of how capacity management affects seasonal pricing strategy.</p>
                </div>
            </div>
        </div>

        <div class="section" id="weekday-weekend">
            <h2>Weekday vs Weekend Analysis</h2>
            <p>Booking patterns show significant differences between weekdays and weekends.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Weekday vs Weekend Price</h3>
                    <img src="../visualizations/weekday_weekend_analysis.png" alt="Weekday vs Weekend Price">
                    <p>Comparison of pricing patterns between weekdays and weekends.</p>
                </div>

                <div class="viz-card">
                    <h3>Weekday vs Weekend Elasticity</h3>
                    <img src="../elasticity_analysis/visualizations/weekend_weekday_comprehensive.png" alt="Weekday vs Weekend Elasticity">
                    <p>How price sensitivity differs between weekday and weekend bookings.</p>
                </div>
            </div>
        </div>

        <div class="section" id="model-performance">
            <h2>Model Performance</h2>
            <p>We evaluated multiple machine learning models to predict booking probability based on listing characteristics.</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Feature Importance: Random Forest</h3>
                    <img src="../models/rf_feature_importance.png" alt="Random Forest Feature Importance">
                    <p>The most influential features in determining booking probability according to our Random Forest model.</p>
                </div>

                <div class="viz-card">
                    <h3>Permutation Importance</h3>
                    <img src="../models/permutation_importance.png" alt="Permutation Importance">
                    <p>Feature importance determined through permutation analysis, showing which features have the greatest impact on model predictions.</p>
                </div>

                <div class="viz-card">
                    <h3>Model Comparison</h3>
                    <img src="../models/rfe_cv_scores.png" alt="Model Comparison">
                    <p>Performance comparison of different machine learning models tested.</p>
                </div>

                <div class="viz-card">
                    <h3>Correlation Matrix</h3>
                    <img src="../visualizations/correlation_analysis.png" alt="Correlation Matrix">
                    <p>Correlation between different listing features, highlighting relationships between variables.</p>
                </div>
            </div>
        </div>
        
        <div class="section" id="recommendations">
            <h2>Recommendations</h2>
            <p>Based on our comprehensive analysis, we've developed the following actionable recommendations for Airbnb hosts to optimize their pricing strategy:</p>
            
            <div class="viz-section">
                <div class="viz-card">
                    <h3>Amenities Impact</h3>
                    <img src="../visualizations/amenities_price_impact.png" alt="Amenities Impact">
                    <p>How different amenities affect optimal pricing and booking probability.</p>
                </div>
                
                <div class="viz-card">
                    <h3>Review Impact</h3>
                    <img src="../visualizations/review_scores_analysis.png" alt="Review Impact">
                    <p>The relationship between review scores and price elasticity, showing how hosts can adjust based on their ratings.</p>
                </div>
                
                <div class="viz-card">
                    <h3>Amenities Frequency</h3>
                    <img src="../visualizations/amenities_frequency.png" alt="Amenities Frequency">
                    <p>Distribution of amenities across listings, highlighting opportunities for differentiation.</p>
                </div>
                
                <div class="viz-card">
                    <h3>Amenities Word Cloud</h3>
                    <img src="../visualizations/amenities_wordcloud.png" alt="Amenities Word Cloud">
                    <p>Visual representation of the most common and impactful amenities in the dataset.</p>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <p>Airbnb Data Analysis | Generated April 19, 2025</p>
        <p>This report was automatically generated as part of the Airbnb Price Elasticity Analysis pipeline.</p>
    </footer> 
    </body>
    </html> 
    """
    
    # Write HTML report to file
    with open(os.path.join(REPORT_PATH, 'final_report.html'), 'w') as f:
        f.write(html_content)
    
    # Return paths and status for task completion tracking
    return {
        'report_path': os.path.join(REPORT_PATH, 'final_report.md'),
        'html_report_path': os.path.join(REPORT_PATH, 'final_report.html'),
        'analysis_completed': True,
        'generation_date': current_date
    }

# Define tasks
load_clean_task = PythonOperator(
    task_id='load_and_clean_data',
    python_callable=load_and_clean_data,
    provide_context=True,
    dag=dag
)

process_reviews_task = PythonOperator(
    task_id='process_reviews_data',
    python_callable=process_reviews_data,
    provide_context=True,
    dag=dag
)

exploratory_analysis_task = PythonOperator(
    task_id='exploratory_analysis',
    python_callable=exploratory_analysis,
    provide_context=True,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag
) 

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag
)
 
price_elasticity_analysis_task = PythonOperator(
    task_id='price_elasticity_analysis',
    python_callable=price_elasticity_analysis,
    provide_context=True,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag
) 
# Define task dependencies
load_clean_task >> process_reviews_task >> exploratory_analysis_task >> feature_engineering_task >> train_models_task >> price_elasticity_analysis_task >> generate_report_task