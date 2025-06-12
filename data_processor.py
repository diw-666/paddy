import pandas as pd
import numpy as np
import os
import re
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class RiceDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.combined_data = None
        
    def extract_season_year(self, filename: str) -> Tuple[str, int]:
        """Extract season and year from filename"""
        # Handle different filename formats
        if "Yala" in filename:
            season = "Yala"
            year_match = re.search(r'(\d{4})', filename)
            year = int(year_match.group(1)) if year_match else None
        elif "Maha" in filename:
            season = "Maha"
            # For Maha season, extract the starting year (e.g., 2022 from "2022-2023 Maha")
            year_matches = re.findall(r'(\d{4})', filename)
            year = int(year_matches[0]) if year_matches else None
        else:
            season = "Unknown"
            year = None
        
        return season, year
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and order across different file formats"""
        # Standard column mapping
        column_mapping = {
            'Major_Schemes_Sown': 'Major_Schemes_Sown',
            'Minor_Schemes_Sown': 'Minor_Schemes_Sown', 
            'Rainfed_Sown': 'Rainfed_Sown',
            'All_Schemes_Sown': 'All_Schemes_Sown',
            'Major_Schemes_Harvested': 'Major_Schemes_Harvested',
            'Minor_Schemes_Harvested': 'Minor_Schemes_Harvested',
            'Rainfed_Harvested': 'Rainfed_Harvested',
            'All_Schemes_Harvested': 'All_Schemes_Harvested',
            'Major_Schemes_Yield': 'Major_Schemes_Yield',
            'Minor_Schemes_Yield': 'Minor_Schemes_Yield',
            'Rainfed_Yield': 'Rainfed_Yield',
            'Average_Yield': 'Average_Yield',
            'Total_Production': 'Total_Production',
            'Nett_Extent_Harvested': 'Nett_Extent_Harvested'
        }
        
        # Rename columns to standard names
        df_standardized = df.copy()
        
        # Calculate missing columns if they don't exist
        if 'All_Schemes_Sown' not in df_standardized.columns:
            df_standardized['All_Schemes_Sown'] = (
                df_standardized.get('Major_Schemes_Sown', 0) + 
                df_standardized.get('Minor_Schemes_Sown', 0) + 
                df_standardized.get('Rainfed_Sown', 0)
            )
        
        if 'All_Schemes_Harvested' not in df_standardized.columns:
            df_standardized['All_Schemes_Harvested'] = (
                df_standardized.get('Major_Schemes_Harvested', 0) + 
                df_standardized.get('Minor_Schemes_Harvested', 0) + 
                df_standardized.get('Rainfed_Harvested', 0)
            )
        
        return df_standardized
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        df_clean = df.copy()
        
        # Replace '-' and empty strings with NaN
        df_clean = df_clean.replace(['-', '', ' '], np.nan)
        
        # Convert numeric columns
        numeric_columns = [col for col in df_clean.columns if col != 'District']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows where District is null or contains 'SRI_LANKA' (total row)
        df_clean = df_clean[df_clean['District'].notna()]
        df_clean = df_clean[~df_clean['District'].str.contains('SRI_LANKA', na=False)]
        
        # Standardize district names
        df_clean['District'] = df_clean['District'].str.upper().str.strip()
        
        return df_clean
    
    def load_and_combine_data(self) -> pd.DataFrame:
        """Load all CSV files and combine into single dataset"""
        all_data = []
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            try:
                # Read CSV file
                filepath = os.path.join(self.data_dir, filename)
                df = pd.read_csv(filepath)
                
                # Extract season and year
                season, year = self.extract_season_year(filename)
                
                if year is None:
                    print(f"Warning: Could not extract year from {filename}")
                    continue
                
                # Standardize columns
                df = self.standardize_columns(df)
                
                # Clean data
                df = self.clean_data(df)
                
                # Add metadata
                df['Season'] = season
                df['Year'] = year
                df['Season_Year'] = f"{season}_{year}"
                
                all_data.append(df)
                print(f"Processed: {filename} - {season} {year} ({len(df)} districts)")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if all_data:
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Sort by year and season
            season_order = {'Yala': 1, 'Maha': 2}
            combined_df['Season_Order'] = combined_df['Season'].map(season_order)
            combined_df = combined_df.sort_values(['Year', 'Season_Order', 'District'])
            combined_df = combined_df.drop('Season_Order', axis=1)
            
            self.combined_data = combined_df
            print(f"\nTotal combined dataset: {len(combined_df)} records")
            print(f"Years covered: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
            print(f"Districts: {combined_df['District'].nunique()}")
            
            return combined_df
        else:
            raise ValueError("No data files could be processed")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for machine learning"""
        df_features = df.copy()
        
        # Fill missing values with 0 for calculation purposes
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        
        # Calculate ratios and efficiency metrics
        df_features['Harvest_Efficiency'] = np.where(
            df_features['All_Schemes_Sown'] > 0,
            df_features['All_Schemes_Harvested'] / df_features['All_Schemes_Sown'],
            0
        )
        
        df_features['Major_Scheme_Ratio'] = np.where(
            df_features['All_Schemes_Sown'] > 0,
            df_features['Major_Schemes_Sown'] / df_features['All_Schemes_Sown'],
            0
        )
        
        df_features['Rainfed_Ratio'] = np.where(
            df_features['All_Schemes_Sown'] > 0,
            df_features['Rainfed_Sown'] / df_features['All_Schemes_Sown'],
            0
        )
        
        # Season encoding
        df_features['Is_Yala'] = (df_features['Season'] == 'Yala').astype(int)
        df_features['Is_Maha'] = (df_features['Season'] == 'Maha').astype(int)
        
        # Time-based features
        df_features['Years_Since_2004'] = df_features['Year'] - 2004
        
        return df_features
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for ML model"""
        return [
            'Major_Schemes_Sown', 'Minor_Schemes_Sown', 'Rainfed_Sown',
            'Major_Schemes_Harvested', 'Minor_Schemes_Harvested', 'Rainfed_Harvested',
            'Harvest_Efficiency', 'Major_Scheme_Ratio', 'Rainfed_Ratio',
            'Is_Yala', 'Is_Maha', 'Years_Since_2004'
        ]
    
    def get_target_columns(self) -> List[str]:
        """Get list of target columns for prediction"""
        return ['Average_Yield', 'Total_Production']
    
    def save_processed_data(self, filepath: str = "processed_rice_data.csv"):
        """Save processed data to CSV"""
        if self.combined_data is not None:
            self.combined_data.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        else:
            print("No data to save. Please load and combine data first.")
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all data and return processed DataFrame"""
        # Check if processed data file exists
        if os.path.exists("processed_rice_data.csv"):
            try:
                df = pd.read_csv("processed_rice_data.csv")
                df = self.create_features(df)
                self.combined_data = df
                return df
            except:
                pass
        
        # Load and process from raw data
        df = self.load_and_combine_data()
        df = self.create_features(df)
        
        # Save processed data for future use
        self.save_processed_data()
        
        return df

if __name__ == "__main__":
    # Test the processor
    processor = RiceDataProcessor()
    data = processor.load_and_combine_data()
    features_data = processor.create_features(data)
    processor.save_processed_data()
    
    print("\nDataset summary:")
    print(features_data.info())
    print("\nFirst few rows:")
    print(features_data.head()) 