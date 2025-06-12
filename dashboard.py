import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processor import RiceDataProcessor
from ml_model import RiceYieldPredictor

# Page configuration
st.set_page_config(
    page_title="Rice Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .anomaly-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .anomaly-normal {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process data with caching"""
    processor = RiceDataProcessor()
    try:
        data = processor.load_and_combine_data()
        features_data = processor.create_features(data)
        return features_data, processor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_resource
def load_model(features_data):
    """Load or train model with caching"""
    predictor = RiceYieldPredictor()
    
    # Try to load existing model
    try:
        predictor.load_models()
        st.success("✅ Pre-trained model loaded successfully!")
    except:
        # Train new model if no saved model exists
        with st.spinner("🔄 Training new model... This may take a few minutes."):
            predictor.train_models(features_data)
            predictor.save_models()
        st.success("✅ New model trained and saved!")
    
    return predictor

def create_historical_chart(data, district, season=None):
    """Create historical yield and production charts"""
    district_data = data[data['District'] == district.upper()]
    
    if season:
        district_data = district_data[district_data['Season'] == season]
    
    if len(district_data) == 0:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Yield Over Time', 'Total Production Over Time'),
        vertical_spacing=0.1
    )
    
    # Yield chart
    fig.add_trace(
        go.Scatter(
            x=district_data['Year'],
            y=district_data['Average_Yield'],
            mode='lines+markers',
            name='Average Yield',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Production chart
    fig.add_trace(
        go.Scatter(
            x=district_data['Year'],
            y=district_data['Total_Production'],
            mode='lines+markers',
            name='Total Production',
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text=f"Historical Trends for {district} {f'({season} Season)' if season else '(All Seasons)'}"
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Yield (kg/ha)", row=1, col=1)
    fig.update_yaxes(title_text="Production (tons)", row=2, col=1)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Yield Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and model
    features_data, processor = load_data()
    
    if features_data is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    predictor = load_model(features_data)
    
    # Sidebar for inputs
    st.sidebar.title("🔧 Prediction Settings")
    
    # Get unique districts and sort them
    districts = sorted(features_data['District'].unique())
    seasons = ['Yala', 'Maha']
    
    # Input widgets
    selected_district = st.sidebar.selectbox(
        "📍 Select District:",
        districts,
        index=0
    )
    
    selected_season = st.sidebar.selectbox(
        "🌱 Select Season:",
        seasons,
        index=0
    )
    
    selected_year = st.sidebar.number_input(
        "📅 Select Year:",
        min_value=2024,
        max_value=2030,
        value=2024,
        step=1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Cultivation Data")
    
    # Get historical averages for the selected district
    district_data = features_data[features_data['District'] == selected_district.upper()]
    
    if len(district_data) > 0:
        avg_major = int(district_data['Major_Schemes_Sown'].mean())
        avg_minor = int(district_data['Minor_Schemes_Sown'].mean())
        avg_rainfed = int(district_data['Rainfed_Sown'].mean())
        harvest_ratio = district_data['Harvest_Efficiency'].mean()
    else:
        avg_major = 20000
        avg_minor = 10000
        avg_rainfed = 5000
        harvest_ratio = 0.9
    
    # Cultivation inputs
    major_sown = st.sidebar.number_input(
        "Major Schemes Sown (ha):",
        min_value=0,
        value=avg_major,
        step=1000
    )
    
    minor_sown = st.sidebar.number_input(
        "Minor Schemes Sown (ha):",
        min_value=0,
        value=avg_minor,
        step=500
    )
    
    rainfed_sown = st.sidebar.number_input(
        "Rainfed Sown (ha):",
        min_value=0,
        value=avg_rainfed,
        step=500
    )
    
    # Calculate totals and harvested areas
    all_schemes_sown = major_sown + minor_sown + rainfed_sown
    
    # Assume typical harvest efficiency
    major_harvested = int(major_sown * harvest_ratio)
    minor_harvested = int(minor_sown * harvest_ratio)
    rainfed_harvested = int(rainfed_sown * harvest_ratio * 0.8)  # Lower efficiency for rainfed
    all_schemes_harvested = major_harvested + minor_harvested + rainfed_harvested
    
    # Display calculated values
    st.sidebar.info(f"""
    **Calculated Values:**
    - Total Sown: {all_schemes_sown:,} ha
    - Total Harvested: {all_schemes_harvested:,} ha
    - Harvest Efficiency: {(all_schemes_harvested/all_schemes_sown)*100:.1f}%
    """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Prediction Results")
        
        # Prediction tabs
        prediction_tab, forecast_tab = st.tabs(["Single Year Prediction", "Multi-Year Forecast"])
        
        with prediction_tab:
            if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
                try:
                    # Make prediction
                    prediction = predictor.predict(
                        district=selected_district,
                        season=selected_season,
                        year=selected_year,
                        Major_Schemes_Sown=major_sown,
                        Minor_Schemes_Sown=minor_sown,
                        Rainfed_Sown=rainfed_sown,
                        Major_Schemes_Harvested=major_harvested,
                        Minor_Schemes_Harvested=minor_harvested,
                        Rainfed_Harvested=rainfed_harvested,
                        All_Schemes_Sown=all_schemes_sown,
                        All_Schemes_Harvested=all_schemes_harvested
                    )
                    
                    # Display results
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(
                            label="🌾 Predicted Average Yield",
                            value=f"{prediction['predicted_yield']:.2f} kg/ha",
                            delta=None
                        )
                    
                    with col_b:
                        st.metric(
                            label="📦 Predicted Total Production", 
                            value=f"{prediction['predicted_production']:,.0f} tons",
                            delta=None
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store prediction in session state for anomaly detection
                    st.session_state['last_prediction'] = prediction
                    
                    # Show success message
                    st.success(f"✅ Prediction completed for {selected_district} - {selected_season} {selected_year}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        with forecast_tab:
            st.markdown("#### 🔮 Multi-Year Forecast")
            
            col_forecast_1, col_forecast_2 = st.columns(2)
            with col_forecast_1:
                forecast_horizon = st.number_input(
                    "Forecast Horizon (years):",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1
                )
            
            with col_forecast_2:
                base_forecast_year = st.number_input(
                    "Base Year:",
                    min_value=2024,
                    max_value=2030,
                    value=2024,
                    step=1
                )
            
            if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
                try:
                    with st.spinner("Generating forecast..."):
                        # Generate forecast
                        forecast_results = predictor.forecast(
                            district=selected_district,
                            season=selected_season,
                            base_year=base_forecast_year,
                            horizon=forecast_horizon,
                            Major_Schemes_Sown=major_sown,
                            Minor_Schemes_Sown=minor_sown,
                            Rainfed_Sown=rainfed_sown,
                            Major_Schemes_Harvested=major_harvested,
                            Minor_Schemes_Harvested=minor_harvested,
                            Rainfed_Harvested=rainfed_harvested,
                            All_Schemes_Sown=all_schemes_sown,
                            All_Schemes_Harvested=all_schemes_harvested
                        )
                    
                    # Create forecast visualization
                    years = [base_forecast_year] + [f['year'] for f in forecast_results]
                    yields = [predictor.predict(
                        district=selected_district,
                        season=selected_season,
                        year=base_forecast_year,
                        Major_Schemes_Sown=major_sown,
                        Minor_Schemes_Sown=minor_sown,
                        Rainfed_Sown=rainfed_sown,
                        Major_Schemes_Harvested=major_harvested,
                        Minor_Schemes_Harvested=minor_harvested,
                        Rainfed_Harvested=rainfed_harvested,
                        All_Schemes_Sown=all_schemes_sown,
                        All_Schemes_Harvested=all_schemes_harvested
                    )['predicted_yield']] + [f['predicted_yield'] for f in forecast_results]
                    
                    productions = [predictor.predict(
                        district=selected_district,
                        season=selected_season,
                        year=base_forecast_year,
                        Major_Schemes_Sown=major_sown,
                        Minor_Schemes_Sown=minor_sown,
                        Rainfed_Sown=rainfed_sown,
                        Major_Schemes_Harvested=major_harvested,
                        Minor_Schemes_Harvested=minor_harvested,
                        Rainfed_Harvested=rainfed_harvested,
                        All_Schemes_Sown=all_schemes_sown,
                        All_Schemes_Harvested=all_schemes_harvested
                    )['predicted_production']] + [f['predicted_production'] for f in forecast_results]
                    
                    # Create forecast chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Yield Forecast', 'Production Forecast'),
                        vertical_spacing=0.15
                    )
                    
                    # Yield forecast
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=yields,
                            mode='lines+markers',
                            name='Predicted Yield',
                            line=dict(color='#2E8B57', width=3),
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )
                    
                    # Add confidence intervals for forecast period
                    if len(forecast_results) > 0:
                        forecast_years = [f['year'] for f in forecast_results]
                        upper_yields = [f['confidence_interval']['upper_yield'] for f in forecast_results]
                        lower_yields = [f['confidence_interval']['lower_yield'] for f in forecast_results]
                        
                        # Upper confidence bound
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_years,
                                y=upper_yields,
                                mode='lines',
                                name='Upper 95% CI',
                                line=dict(color='rgba(46, 139, 87, 0.3)', dash='dash'),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        # Lower confidence bound
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_years,
                                y=lower_yields,
                                mode='lines',
                                name='Lower 95% CI',
                                line=dict(color='rgba(46, 139, 87, 0.3)', dash='dash'),
                                fill='tonexty',
                                fillcolor='rgba(46, 139, 87, 0.1)',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    # Production forecast
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=productions,
                            mode='lines+markers',
                            name='Predicted Production',
                            line=dict(color='#FF6B35', width=3),
                            marker=dict(size=8)
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=600,
                        title_text=f"Multi-Year Forecast: {selected_district} - {selected_season} Season",
                        showlegend=True
                    )
                    
                    fig.update_xaxes(title_text="Year")
                    fig.update_yaxes(title_text="Yield (kg/ha)", row=1, col=1)
                    fig.update_yaxes(title_text="Production (tons)", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary table
                    st.markdown("#### 📊 Forecast Details")
                    
                    # Create summary dataframe
                    forecast_df = pd.DataFrame([
                        {
                            'Year': base_forecast_year,
                            'Predicted Yield (kg/ha)': f"{yields[0]:.1f}",
                            'Predicted Production (tons)': f"{productions[0]:,.0f}",
                            'Change from Previous': "Baseline"
                        }
                    ] + [
                        {
                            'Year': f['year'],
                            'Predicted Yield (kg/ha)': f"{f['predicted_yield']:.1f}",
                            'Predicted Production (tons)': f"{f['predicted_production']:,.0f}",
                            'Change from Previous': f"{f['yield_change_from_previous']:+.1f} kg/ha"
                        }
                        for f in forecast_results
                    ])
                    
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                    
                    # Historical Analysis Summary
                    if forecast_results and 'historical_context' in forecast_results[0]:
                        st.markdown("#### 🔍 Historical Analysis Used")
                        
                        context = forecast_results[0]['historical_context']
                        col_hist1, col_hist2, col_hist3, col_hist4 = st.columns(4)
                        
                        with col_hist1:
                            st.metric("Years of Data", f"{context['years_of_data']}")
                        
                        with col_hist2:
                            st.metric("Historical Std Dev", f"{context['historical_std']:.1f} kg/ha")
                        
                        with col_hist3:
                            st.metric("Trend Type", context['trend_type'].title())
                        
                        with col_hist4:
                            st.metric("Trend Slope", f"{context['trend_slope']:+.1f} kg/ha/yr")
                        
                        # Show forecast components for the first forecast year
                        if 'components' in forecast_results[0]:
                            st.markdown("#### ⚙️ Forecast Components (Year 1)")
                            components = forecast_results[0]['components']
                            
                            components_df = pd.DataFrame([
                                {'Component': 'Base Prediction', 'Value (kg/ha)': f"{components['base_prediction']:.1f}"},
                                {'Component': 'Trend Adjustment', 'Value (kg/ha)': f"{components['trend']:+.1f}"},
                                {'Component': 'Cyclical Effect', 'Value (kg/ha)': f"{components['cyclical']:+.1f}"},
                                {'Component': 'Seasonal Effect', 'Value (kg/ha)': f"{components['seasonal']:+.1f}"},
                                {'Component': 'Autocorrelation', 'Value (kg/ha)': f"{components['autocorr']:+.1f}"},
                                {'Component': 'Weather Effect', 'Value (kg/ha)': f"{components['weather']:+.1f}"},
                                {'Component': 'Long-term Variability', 'Value (kg/ha)': f"{components['longterm']:+.1f}"},
                                {'Component': 'Total Adjustment', 'Value (kg/ha)': f"{components['total_adjustment']:+.1f}"},
                            ])
                            
                            st.dataframe(components_df, use_container_width=True, hide_index=True)
                    
                    # Store forecast in session state
                    st.session_state['last_forecast'] = forecast_results
                    
                    # Summary statistics
                    yield_values = [f['predicted_yield'] for f in forecast_results]
                    if yield_values:
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            st.metric(
                                "Avg Annual Change",
                                f"{sum(f['yield_change_from_previous'] for f in forecast_results) / len(forecast_results):+.1f} kg/ha"
                            )
                        
                        with col_stats2:
                            st.metric(
                                "Yield Range",
                                f"{min(yield_values):.0f} - {max(yield_values):.0f} kg/ha"
                            )
                        
                        with col_stats3:
                            st.metric(
                                "Forecast Variability",
                                f"{((max(yield_values) - min(yield_values)) / min(yield_values) * 100):.1f}%"
                            )
                    
                    st.success(f"✅ Forecast completed for {forecast_horizon} years!")
                    
                except Exception as e:
                    st.error(f"❌ Forecast failed: {str(e)}")
                    st.exception(e)  # For debugging
    
    with col2:
        st.markdown("### 🔍 Anomaly Detection")
        
        # Show anomaly detection form if prediction exists
        if 'last_prediction' in st.session_state:
            prediction = st.session_state['last_prediction']
            
            st.markdown("**Compare with actual values:**")
            
            actual_yield = st.number_input(
                "Actual Yield (kg/ha):",
                min_value=0.0,
                value=prediction['predicted_yield'],
                step=0.1,
                format="%.2f"
            )
            
            actual_production = st.number_input(
                "Actual Production (tons):",
                min_value=0.0,
                value=prediction['predicted_production'],
                step=100.0,
                format="%.0f"
            )
            
            threshold = st.slider(
                "Anomaly Threshold:",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                format="%.2f",
                help="Percentage deviation to flag as anomaly"
            )
            
            if st.button("🔍 Check for Anomalies", use_container_width=True):
                anomaly_result = predictor.detect_anomaly(
                    district=selected_district,
                    actual_yield=actual_yield,
                    actual_production=actual_production,
                    predicted_yield=prediction['predicted_yield'],
                    predicted_production=prediction['predicted_production'],
                    threshold=threshold
                )
                
                # Display anomaly results
                if anomaly_result['is_anomaly']:
                    st.markdown(f'<div class="anomaly-warning">{anomaly_result["message"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="anomaly-normal">{anomaly_result["message"]}</div>', 
                              unsafe_allow_html=True)
                
                # Show deviations
                st.markdown("**Deviation Analysis:**")
                st.write(f"• Yield Deviation: {anomaly_result['yield_deviation']:.1%}")
                st.write(f"• Production Deviation: {anomaly_result['production_deviation']:.1%}")
        else:
            st.info("🎯 Make a prediction first to enable anomaly detection")
    
    # Historical Data Section
    st.markdown("---")
    st.markdown("### 📈 Historical Analysis")
    
    col3, col4 = st.columns([3, 1])
    
    with col4:
        show_season_filter = st.checkbox("Filter by season", value=False)
        chart_season = selected_season if show_season_filter else None
        
        # District statistics
        district_stats = district_data.groupby('Season').agg({
            'Average_Yield': ['mean', 'std'],
            'Total_Production': ['mean', 'std']
        }).round(2)
        
        if len(district_stats) > 0:
            st.markdown("**District Statistics:**")
            st.dataframe(district_stats, use_container_width=True)
    
    with col3:
        # Create and display historical chart
        historical_chart = create_historical_chart(features_data, selected_district, chart_season)
        
        if historical_chart:
            st.plotly_chart(historical_chart, use_container_width=True)
        else:
            st.warning(f"No historical data available for {selected_district}")
    
    # Data Overview Section
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Total Records", f"{len(features_data):,}")
    
    with col6:
        st.metric("Districts", f"{features_data['District'].nunique()}")
    
    with col7:
        st.metric("Years Covered", f"{features_data['Year'].nunique()}")
    
    with col8:
        year_range = f"{features_data['Year'].min()}-{features_data['Year'].max()}"
        st.metric("Year Range", year_range)
    
    # Show recent data table
    if st.checkbox("Show Recent Data", value=False):
        recent_data = features_data.sort_values(['Year', 'District']).tail(50)
        display_columns = ['District', 'Season', 'Year', 'Average_Yield', 'Total_Production', 
                          'All_Schemes_Sown', 'All_Schemes_Harvested']
        st.dataframe(recent_data[display_columns], use_container_width=True)

if __name__ == "__main__":
    main() 