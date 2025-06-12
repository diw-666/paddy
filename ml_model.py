import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class RiceYieldPredictor:
    def __init__(self):
        self.yield_model = None
        self.production_model = None
        self.scaler = None
        self.district_encoder = None
        self.feature_columns = None
        self.district_stats = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for machine learning"""
        df_features = df.copy()
        
        # Encode districts
        if self.district_encoder is None:
            self.district_encoder = LabelEncoder()
            df_features['District_Encoded'] = self.district_encoder.fit_transform(df_features['District'])
        else:
            # Handle new districts during prediction
            df_features['District_Encoded'] = df_features['District'].apply(
                lambda x: self.district_encoder.transform([x])[0] if x in self.district_encoder.classes_ else -1
            )
        
        # Select feature columns
        feature_cols = [
            'Major_Schemes_Sown', 'Minor_Schemes_Sown', 'Rainfed_Sown',
            'Major_Schemes_Harvested', 'Minor_Schemes_Harvested', 'Rainfed_Harvested',
            'Harvest_Efficiency', 'Major_Scheme_Ratio', 'Rainfed_Ratio',
            'Is_Yala', 'Is_Maha', 'Years_Since_2004', 'District_Encoded'
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        
        X = df_features[feature_cols].fillna(0)
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Prepare targets (only during training)
        if is_training:
            y_yield = df_features['Average_Yield'].fillna(0).values
            y_production = df_features['Total_Production'].fillna(0).values
        else:
            y_yield = np.zeros(len(df_features))
            y_production = np.zeros(len(df_features))
        
        self.feature_columns = feature_cols
        
        return X_scaled, y_yield, y_production
    
    def train_models(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Train prediction models"""
        print("Preparing features...")
        X, y_yield, y_production = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_yield_train, y_yield_test, y_prod_train, y_prod_test = train_test_split(
            X, y_yield, y_production, test_size=test_size, random_state=random_state
        )
        
        print("Training yield prediction model...")
        self.yield_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.yield_model.fit(X_train, y_yield_train)
        
        print("Training production prediction model...")
        self.production_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        self.production_model.fit(X_train, y_prod_train)
        
        # Evaluate models
        print("\nModel Evaluation:")
        self._evaluate_model(X_test, y_yield_test, y_prod_test)
        
        # Calculate district statistics for anomaly detection
        self._calculate_district_stats(df)
        
        self.is_trained = True
        print("Models trained successfully!")
    
    def _evaluate_model(self, X_test, y_yield_test, y_prod_test):
        """Evaluate model performance"""
        # Yield predictions
        yield_pred = self.yield_model.predict(X_test)
        yield_mae = mean_absolute_error(y_yield_test, yield_pred)
        yield_rmse = np.sqrt(mean_squared_error(y_yield_test, yield_pred))
        yield_r2 = r2_score(y_yield_test, yield_pred)
        
        # Production predictions  
        prod_pred = self.production_model.predict(X_test)
        prod_mae = mean_absolute_error(y_prod_test, prod_pred)
        prod_rmse = np.sqrt(mean_squared_error(y_prod_test, prod_pred))
        prod_r2 = r2_score(y_prod_test, prod_pred)
        
        print(f"Yield Model - MAE: {yield_mae:.2f}, RMSE: {yield_rmse:.2f}, R²: {yield_r2:.3f}")
        print(f"Production Model - MAE: {prod_mae:.0f}, RMSE: {prod_rmse:.0f}, R²: {prod_r2:.3f}")
    
    def _calculate_district_stats(self, df: pd.DataFrame):
        """Calculate historical statistics for each district for anomaly detection"""
        self.district_stats = {}
        
        for district in df['District'].unique():
            district_data = df[df['District'] == district]
            
            self.district_stats[district] = {
                'yield_mean': district_data['Average_Yield'].mean(),
                'yield_std': district_data['Average_Yield'].std(),
                'production_mean': district_data['Total_Production'].mean(),
                'production_std': district_data['Total_Production'].std(),
                'yield_min': district_data['Average_Yield'].min(),
                'yield_max': district_data['Average_Yield'].max(),
                'production_min': district_data['Total_Production'].min(),
                'production_max': district_data['Total_Production'].max()
            }
    
    def predict(self, district: str, season: str, year: int, **kwargs) -> Dict[str, Any]:
        """Make predictions for a specific district, season, and year"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Create input data
        input_data = {
            'District': district.upper(),
            'Season': season,
            'Year': year,
            'Is_Yala': 1 if season == 'Yala' else 0,
            'Is_Maha': 1 if season == 'Maha' else 0,
            'Years_Since_2004': year - 2004,
            **kwargs
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Calculate derived features
        input_df['Harvest_Efficiency'] = np.where(
            input_df.get('All_Schemes_Sown', 0) > 0,
            input_df.get('All_Schemes_Harvested', 0) / input_df.get('All_Schemes_Sown', 1),
            0.9  # Default efficiency
        )
        
        input_df['Major_Scheme_Ratio'] = np.where(
            input_df.get('All_Schemes_Sown', 0) > 0,
            input_df.get('Major_Schemes_Sown', 0) / input_df.get('All_Schemes_Sown', 1),
            0.5  # Default ratio
        )
        
        input_df['Rainfed_Ratio'] = np.where(
            input_df.get('All_Schemes_Sown', 0) > 0,
            input_df.get('Rainfed_Sown', 0) / input_df.get('All_Schemes_Sown', 1),
            0.3  # Default ratio
        )
        
        # Prepare features
        X, _, _ = self.prepare_features(input_df, is_training=False)
        
        # Make predictions
        predicted_yield = self.yield_model.predict(X)[0]
        predicted_production = self.production_model.predict(X)[0]
        
        # Ensure non-negative predictions
        predicted_yield = max(0, predicted_yield)
        predicted_production = max(0, predicted_production)
        
        return {
            'predicted_yield': predicted_yield,
            'predicted_production': predicted_production,
            'district': district,
            'season': season,
            'year': year
        }
    
    def detect_anomaly(self, district: str, actual_yield: float, actual_production: float, 
                      predicted_yield: float, predicted_production: float, 
                      threshold: float = 0.3) -> Dict[str, Any]:
        """Detect anomalies by comparing actual vs predicted values"""
        
        anomaly_info = {
            'is_anomaly': False,
            'anomaly_type': 'Normal',
            'yield_deviation': 0,
            'production_deviation': 0,
            'message': '✅ Normal - Values within expected range'
        }
        
        # Calculate percentage deviations
        if predicted_yield > 0:
            yield_deviation = abs(actual_yield - predicted_yield) / predicted_yield
        else:
            yield_deviation = 0
            
        if predicted_production > 0:
            production_deviation = abs(actual_production - predicted_production) / predicted_production
        else:
            production_deviation = 0
        
        anomaly_info['yield_deviation'] = yield_deviation
        anomaly_info['production_deviation'] = production_deviation
        
        # Check for anomalies
        if yield_deviation > threshold or production_deviation > threshold:
            anomaly_info['is_anomaly'] = True
            
            # Determine anomaly type
            if actual_yield < predicted_yield * (1 - threshold):
                anomaly_info['anomaly_type'] = 'Low Yield'
                anomaly_info['message'] = f'⚠️ Anomaly: Yield {yield_deviation:.1%} lower than expected'
            elif actual_yield > predicted_yield * (1 + threshold):
                anomaly_info['anomaly_type'] = 'High Yield'
                anomaly_info['message'] = f'📈 Anomaly: Yield {yield_deviation:.1%} higher than expected'
            elif actual_production < predicted_production * (1 - threshold):
                anomaly_info['anomaly_type'] = 'Low Production'
                anomaly_info['message'] = f'⚠️ Anomaly: Production {production_deviation:.1%} lower than expected'
            elif actual_production > predicted_production * (1 + threshold):
                anomaly_info['anomaly_type'] = 'High Production'
                anomaly_info['message'] = f'📈 Anomaly: Production {production_deviation:.1%} higher than expected'
        
        # Additional historical context
        if district.upper() in self.district_stats:
            stats = self.district_stats[district.upper()]
            
            # Check against historical bounds
            if actual_yield < stats['yield_min'] or actual_yield > stats['yield_max']:
                anomaly_info['is_anomaly'] = True
                anomaly_info['message'] += f' (Outside historical range: {stats["yield_min"]:.1f}-{stats["yield_max"]:.1f})'
        
        return anomaly_info
    
    def forecast(self, district: str, season: str, base_year: int, horizon: int = 5, **base_params) -> List[Dict[str, Any]]:
        """Generate realistic multi-year forecast using comprehensive historical analysis"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        import numpy as np
        from data_processor import RiceDataProcessor
        from scipy import stats
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        forecast_results = []
        
        # Get comprehensive historical data for this district
        processor = RiceDataProcessor()
        data = processor.load_and_combine_data()
        features_data = processor.create_features(data)
        
        district_history = features_data[
            (features_data['District'] == district.upper()) & 
            (features_data['Season'] == season)
        ].sort_values('Year')
        
        if len(district_history) > 0:
            print(f"Using {len(district_history)} years of historical data ({district_history['Year'].min()}-{district_history['Year'].max()}) for {district} {season} season")
            
            # === COMPREHENSIVE HISTORICAL ANALYSIS ===
            years = district_history['Year'].values
            yields = district_history['Average_Yield'].values
            productions = district_history['Total_Production'].values
            
            # 1. Long-term trend analysis (using ALL available data)
            if len(district_history) >= 3:
                # Linear trend
                linear_trend_slope, linear_intercept, r_value, p_value, std_err = stats.linregress(years, yields)
                
                # Polynomial trend (quadratic) for non-linear patterns
                poly_features = PolynomialFeatures(degree=2)
                years_poly = poly_features.fit_transform(years.reshape(-1, 1))
                poly_model = LinearRegression().fit(years_poly, yields)
                
                # Use polynomial if it fits significantly better
                linear_r2 = r_value ** 2
                poly_predictions = poly_model.predict(years_poly)
                poly_r2 = 1 - np.sum((yields - poly_predictions) ** 2) / np.sum((yields - np.mean(yields)) ** 2)
                
                use_polynomial = poly_r2 > linear_r2 + 0.05  # Use poly if 5% better
                
                print(f"Trend analysis: Linear R²={linear_r2:.3f}, Polynomial R²={poly_r2:.3f}, Using {'polynomial' if use_polynomial else 'linear'}")
            else:
                linear_trend_slope = 0
                linear_intercept = np.mean(yields) if len(yields) > 0 else 3500
                use_polynomial = False
                poly_model = None
                poly_features = None
            
            # 2. Cyclical pattern analysis (multi-year cycles)
            yield_detrended = yields - (linear_trend_slope * years + linear_intercept)
            cyclical_pattern = {}
            
            # Look for 3-7 year cycles (common in agriculture due to weather patterns)
            for cycle_length in range(3, min(8, len(district_history)//2)):
                if len(district_history) >= cycle_length * 2:
                    cycle_values = []
                    for i in range(cycle_length):
                        cycle_positions = [j for j in range(i, len(yield_detrended), cycle_length)]
                        if len(cycle_positions) >= 2:
                            cycle_values.append(np.mean([yield_detrended[pos] for pos in cycle_positions]))
                        else:
                            cycle_values.append(0)
                    cyclical_pattern[cycle_length] = cycle_values
            
            # 3. Variability analysis from ALL years
            yield_std = np.std(yields)
            prod_std = np.std(productions)
            
            # Calculate different types of variability
            short_term_var = np.std(np.diff(yields))  # Year-to-year variation
            long_term_var = np.std(yield_detrended)   # Variation around trend
            
            # 4. Autocorrelation analysis
            autocorr_1yr = np.corrcoef(yields[:-1], yields[1:])[0, 1] if len(yields) > 1 else 0
            autocorr_2yr = np.corrcoef(yields[:-2], yields[2:])[0, 1] if len(yields) > 2 else 0
            
            # 5. Seasonal effects (comparing this season with others)
            all_district_data = features_data[features_data['District'] == district.upper()]
            seasonal_effects = {}
            for s in ['Yala', 'Maha']:
                season_data = all_district_data[all_district_data['Season'] == s]
                if len(season_data) > 0:
                    seasonal_effects[s] = np.mean(season_data['Average_Yield'])
            
            season_adjustment = 0
            if len(seasonal_effects) > 1:
                overall_mean = np.mean(list(seasonal_effects.values()))
                season_adjustment = seasonal_effects.get(season, overall_mean) - overall_mean
            
            # 6. Parameter trend analysis
            param_trends = {}
            for param in ['Major_Schemes_Sown', 'Minor_Schemes_Sown', 'Rainfed_Sown', 'Harvest_Efficiency']:
                if param in district_history.columns and len(district_history) >= 3:
                    param_values = district_history[param].values
                    if not np.all(np.isnan(param_values)):
                        param_slope, _, _, _, _ = stats.linregress(years, param_values)
                        param_trends[param] = param_slope
                    else:
                        param_trends[param] = 0
                else:
                    param_trends[param] = 0
            
        else:
            # Default values for districts with no history
            print(f"No historical data found for {district} {season} season, using default values")
            linear_trend_slope = 0
            linear_intercept = 3500
            use_polynomial = False
            poly_model = None
            poly_features = None
            yield_std = 300
            prod_std = 8000
            short_term_var = 200
            long_term_var = 250
            autocorr_1yr = 0.3
            autocorr_2yr = 0.1
            season_adjustment = 0
            cyclical_pattern = {}
            param_trends = {}
        
        # Get base prediction using historical context
        base_prediction = self.predict(
            district=district,
            season=season, 
            year=base_year,
            **base_params
        )
        
        prev_yield = base_prediction['predicted_yield']
        prev_production = base_prediction['predicted_production']
        
        # Store base year result
        base_forecast = {
            'year': base_year,
            'predicted_yield': prev_yield,
            'predicted_production': prev_production,
            'confidence_interval': {
                'lower_yield': max(0, prev_yield - 1.96 * yield_std * 0.2),
                'upper_yield': prev_yield + 1.96 * yield_std * 0.2,
                'lower_production': max(0, prev_production * 0.8),
                'upper_production': prev_production * 1.2
            },
            'yield_change_from_previous': 0,
            'production_change_from_previous': 0,
            'components': {
                'base_prediction': prev_yield,
                'trend': 0,
                'cyclical': 0,
                'seasonal': 0,
                'autocorr': 0,
                'random': 0
            }
        }
        
        # Generate forecast for each future year
        for i in range(1, horizon + 1):
            year = base_year + i
            
            # === PARAMETER EVOLUTION BASED ON HISTORICAL TRENDS ===
            varied_params = base_params.copy()
            
            # Apply historical parameter trends
            for param, trend in param_trends.items():
                if param in varied_params:
                    historical_change = trend * i
                    # Add some noise around the trend
                    noise = np.random.normal(0, abs(trend) * 0.5) if trend != 0 else np.random.normal(0, varied_params[param] * 0.02)
                    new_value = varied_params[param] + historical_change + noise
                    
                    # Apply reasonable bounds
                    if param == 'Harvest_Efficiency':
                        new_value = max(0.6, min(0.98, new_value))
                    else:
                        new_value = max(0, new_value)
                    
                    varied_params[param] = int(new_value) if param != 'Harvest_Efficiency' else new_value
            
            # Additional random variations in cultivation areas
            area_variation = 1 + np.random.normal(0, 0.015)
            area_variation = max(0.9, min(1.1, area_variation))
            
            for key in ['Major_Schemes_Sown', 'Minor_Schemes_Sown', 'Rainfed_Sown']:
                if key in varied_params:
                    varied_params[key] = int(varied_params[key] * area_variation)
            
            # Recalculate dependent parameters
            if 'All_Schemes_Sown' in base_params:
                varied_params['All_Schemes_Sown'] = (
                    varied_params.get('Major_Schemes_Sown', 0) +
                    varied_params.get('Minor_Schemes_Sown', 0) +
                    varied_params.get('Rainfed_Sown', 0)
                )
            
            # Harvest efficiency variations by scheme type
            total_harvested = 0
            for sown_key, harvest_key in [
                ('Major_Schemes_Sown', 'Major_Schemes_Harvested'),
                ('Minor_Schemes_Sown', 'Minor_Schemes_Harvested'),
                ('Rainfed_Sown', 'Rainfed_Harvested')
            ]:
                if sown_key in varied_params:
                    if 'Major' in sown_key:
                        efficiency = np.random.normal(0.92, 0.04)
                    elif 'Minor' in sown_key:
                        efficiency = np.random.normal(0.90, 0.05)
                    else:  # Rainfed
                        efficiency = np.random.normal(0.85, 0.07)
                    
                    efficiency = max(0.6, min(0.98, efficiency))
                    if harvest_key in varied_params or harvest_key in base_params:
                        varied_params[harvest_key] = int(varied_params[sown_key] * efficiency)
                        total_harvested += varied_params[harvest_key]
            
            if 'All_Schemes_Harvested' in base_params:
                varied_params['All_Schemes_Harvested'] = total_harvested
            
            # === GET BASE PREDICTION FOR THIS YEAR ===
            prediction = self.predict(
                district=district,
                season=season,
                year=year,
                **varied_params
            )
            
            # === APPLY COMPREHENSIVE HISTORICAL ADJUSTMENTS ===
            
            # 1. Trend component
            if use_polynomial and poly_model is not None:
                # Use polynomial trend
                year_poly = poly_features.transform([[year]])
                trend_value = poly_model.predict(year_poly)[0]
                trend_adjustment = trend_value - (linear_trend_slope * base_year + linear_intercept)
            else:
                # Use linear trend
                trend_adjustment = linear_trend_slope * i
            
            # 2. Cyclical component (average of detected cycles)
            cyclical_adjustment = 0
            if cyclical_pattern:
                cycle_effects = []
                for cycle_length, cycle_values in cyclical_pattern.items():
                    cycle_position = (year - district_history['Year'].min()) % cycle_length
                    if cycle_position < len(cycle_values):
                        cycle_effects.append(cycle_values[cycle_position])
                
                if cycle_effects:
                    cyclical_adjustment = np.mean(cycle_effects)
            
            # 3. Seasonal adjustment
            seasonal_adjustment = season_adjustment * 0.1  # Scaled down as it's already in the model
            
            # 4. Autocorrelation effects
            autocorr_adjustment = 0
            if i == 1:
                # First year: use 1-year autocorrelation
                autocorr_adjustment = autocorr_1yr * np.random.normal(0, short_term_var * 0.3)
            elif i == 2:
                # Second year: use 2-year autocorrelation  
                autocorr_adjustment = autocorr_2yr * np.random.normal(0, short_term_var * 0.2)
            else:
                # Later years: diminishing effect
                autocorr_adjustment = (autocorr_1yr * 0.5) * np.random.normal(0, short_term_var * 0.1)
            
            # 5. Random weather/environmental effects
            weather_effect = np.random.normal(0, long_term_var * 0.3)
            
            # 6. Long-term variability
            longterm_effect = np.random.normal(0, yield_std * 0.1)
            
            # === COMBINE ALL EFFECTS ===
            total_adjustment = (
                trend_adjustment +
                cyclical_adjustment +
                seasonal_adjustment +
                autocorr_adjustment +
                weather_effect +
                longterm_effect
            )
            
            adjusted_yield = prediction['predicted_yield'] + total_adjustment
            adjusted_yield = max(100, adjusted_yield)  # Minimum reasonable yield
            
            # Adjust production proportionally
            yield_ratio = adjusted_yield / prediction['predicted_yield'] if prediction['predicted_yield'] > 0 else 1
            adjusted_production = prediction['predicted_production'] * yield_ratio
            
            # === CONFIDENCE INTERVALS BASED ON HISTORICAL VARIABILITY ===
            # Use historical standard deviation for confidence intervals
            confidence_width = 1.96 * yield_std * (0.2 + 0.05 * i)  # Increasing uncertainty over time
            confidence_lower = adjusted_yield - confidence_width
            confidence_upper = adjusted_yield + confidence_width
            
            # Store detailed results
            forecast_results.append({
                'year': year,
                'predicted_yield': adjusted_yield,
                'predicted_production': adjusted_production,
                'confidence_interval': {
                    'lower_yield': max(0, confidence_lower),
                    'upper_yield': confidence_upper,
                    'lower_production': max(0, adjusted_production * (confidence_lower / adjusted_yield)) if adjusted_yield > 0 else 0,
                    'upper_production': adjusted_production * (confidence_upper / adjusted_yield) if adjusted_yield > 0 else adjusted_production * 1.2
                },
                'yield_change_from_previous': adjusted_yield - prev_yield,
                'production_change_from_previous': adjusted_production - prev_production,
                'parameters_used': varied_params.copy(),
                'components': {
                    'base_prediction': prediction['predicted_yield'],
                    'trend': trend_adjustment,
                    'cyclical': cyclical_adjustment,
                    'seasonal': seasonal_adjustment,
                    'autocorr': autocorr_adjustment,
                    'weather': weather_effect,
                    'longterm': longterm_effect,
                    'total_adjustment': total_adjustment
                },
                'historical_context': {
                    'years_of_data': len(district_history),
                    'historical_std': yield_std,
                    'trend_slope': linear_trend_slope,
                    'autocorr_1yr': autocorr_1yr,
                    'trend_type': 'polynomial' if use_polynomial else 'linear'
                }
            })
            
            prev_yield = adjusted_yield
            prev_production = adjusted_production
        
        return forecast_results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained models"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        # Yield model importance
        for i, col in enumerate(self.feature_columns):
            importance_dict[f"yield_{col}"] = self.yield_model.feature_importances_[i]
        
        # Production model importance  
        for i, col in enumerate(self.feature_columns):
            importance_dict[f"production_{col}"] = self.production_model.feature_importances_[i]
        
        return importance_dict
    
    def save_models(self, filepath: str = "rice_yield_models.pkl"):
        """Save trained models to file"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'yield_model': self.yield_model,
            'production_model': self.production_model,
            'scaler': self.scaler,
            'district_encoder': self.district_encoder,
            'feature_columns': self.feature_columns,
            'district_stats': self.district_stats,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = "rice_yield_models.pkl"):
        """Load trained models from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        
        self.yield_model = model_data['yield_model']
        self.production_model = model_data['production_model']
        self.scaler = model_data['scaler']
        self.district_encoder = model_data['district_encoder']
        self.feature_columns = model_data['feature_columns']
        self.district_stats = model_data['district_stats']
        self.is_trained = model_data['is_trained']
        
        print(f"Models loaded from {filepath}")

if __name__ == "__main__":
    # Test the model
    from data_processor import RiceDataProcessor
    
    # Load and process data
    processor = RiceDataProcessor()
    data = processor.load_and_combine_data()
    features_data = processor.create_features(data)
    
    # Train model
    predictor = RiceYieldPredictor()
    predictor.train_models(features_data)
    
    # Test prediction
    test_prediction = predictor.predict(
        district="COLOMBO",
        season="Yala", 
        year=2024,
        Major_Schemes_Sown=25000,
        Minor_Schemes_Sown=10000,
        Rainfed_Sown=8000,
        Major_Schemes_Harvested=23000,
        Minor_Schemes_Harvested=9500,
        Rainfed_Harvested=7000,
        All_Schemes_Sown=43000,
        All_Schemes_Harvested=39500
    )
    
    print("\nTest Prediction:")
    print(f"Predicted Yield: {test_prediction['predicted_yield']:.2f} kg/ha")
    print(f"Predicted Production: {test_prediction['predicted_production']:.0f} tons")
    
    # Save models
    predictor.save_models() 