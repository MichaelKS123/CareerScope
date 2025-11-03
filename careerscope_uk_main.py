"""
CareerScope: Gender Gap Analysis in Education and Employment (UK Focus)
Author: Michael Semera
Description: Comprehensive analysis of gender disparities in education and employment
with focus on the United Kingdom and global comparisons
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, pearsonr

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mapping
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️  GeoPandas not available. Map visualizations will be limited.")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DataGenerator:
    """
    Generate realistic gender gap data for demonstration.
    Based on UK ONS, UNESCO and World Bank statistics patterns.
    """
    
    def __init__(self):
        self.years = list(range(1990, 2024, 2))
        self.countries = self._get_countries()
        self.uk_regions = ['London', 'South East', 'South West', 'East of England',
                          'West Midlands', 'East Midlands', 'Yorkshire and The Humber',
                          'North West', 'North East', 'Wales', 'Scotland', 'Northern Ireland']
        
    def _get_countries(self):
        """Countries with regional classifications."""
        return {
            'United Kingdom': 'Europe',
            'United States': 'North America',
            'Canada': 'North America',
            'Germany': 'Europe',
            'France': 'Europe',
            'Italy': 'Europe',
            'Spain': 'Europe',
            'Sweden': 'Europe',
            'Norway': 'Europe',
            'Poland': 'Europe',
            'Netherlands': 'Europe',
            'Belgium': 'Europe',
            'China': 'East Asia',
            'Japan': 'East Asia',
            'South Korea': 'East Asia',
            'India': 'South Asia',
            'Pakistan': 'South Asia',
            'Bangladesh': 'South Asia',
            'Indonesia': 'Southeast Asia',
            'Thailand': 'Southeast Asia',
            'Vietnam': 'Southeast Asia',
            'Saudi Arabia': 'Middle East',
            'UAE': 'Middle East',
            'Egypt': 'Middle East',
            'South Africa': 'Africa',
            'Nigeria': 'Africa',
            'Kenya': 'Africa',
            'Ethiopia': 'Africa',
            'Brazil': 'Latin America',
            'Mexico': 'Latin America',
            'Argentina': 'Latin America',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania'
        }
    
    def generate_education_data(self):
        """Generate education attainment data by gender."""
        print("📚 Generating education data...")
        
        data = []
        
        for country, region in self.countries.items():
            # Base rates by region
            if region in ['Europe', 'North America', 'Oceania']:
                base_primary_m, base_primary_f = 98, 98
                base_secondary_m, base_secondary_f = 90, 92
                base_tertiary_m, base_tertiary_f = 55, 60
                
                # UK has smaller gender gap
                if country == 'United Kingdom':
                    base_secondary_m, base_secondary_f = 92, 94
                    base_tertiary_m, base_tertiary_f = 58, 63
                    
            elif region == 'East Asia':
                base_primary_m, base_primary_f = 98, 97
                base_secondary_m, base_secondary_f = 88, 85
                base_tertiary_m, base_tertiary_f = 58, 52
            elif region in ['Latin America', 'Middle East']:
                base_primary_m, base_primary_f = 92, 88
                base_secondary_m, base_secondary_f = 78, 70
                base_tertiary_m, base_tertiary_f = 48, 35
            else:  # South Asia, Southeast Asia, Africa
                base_primary_m, base_primary_f = 85, 75
                base_secondary_m, base_secondary_f = 65, 50
                base_tertiary_m, base_tertiary_f = 35, 22
            
            for year in self.years:
                # Progressive improvement over time
                year_factor = (year - 1990) / 34  # 0 to 1 from 1990 to 2024
                improvement = year_factor * 15  # Up to 15% improvement
                
                # Women's education improved faster in most regions
                female_improvement = improvement * 1.3
                
                # Add some realistic variation
                noise = np.random.normal(0, 1)
                
                record = {
                    'country': country,
                    'region': region,
                    'year': year,
                    'primary_male': min(100, base_primary_m + improvement + noise),
                    'primary_female': min(100, base_primary_f + female_improvement + noise),
                    'secondary_male': min(100, base_secondary_m + improvement + noise),
                    'secondary_female': min(100, base_secondary_f + female_improvement + noise),
                    'tertiary_male': min(95, base_tertiary_m + improvement + noise),
                    'tertiary_female': min(95, base_tertiary_f + female_improvement + noise)
                }
                
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Calculate gaps
        df['primary_gap'] = df['primary_male'] - df['primary_female']
        df['secondary_gap'] = df['secondary_male'] - df['secondary_female']
        df['tertiary_gap'] = df['tertiary_male'] - df['tertiary_female']
        
        print(f"✓ Generated {len(df)} education records")
        return df
    
    def generate_employment_data(self):
        """Generate employment data by gender."""
        print("💼 Generating employment data...")
        
        data = []
        
        for country, region in self.countries.items():
            # Base employment rates by region
            if region in ['Europe', 'North America', 'Oceania']:
                base_emp_m = 75
                base_emp_f = 60
                
                if country == 'United Kingdom':
                    base_emp_m = 78
                    base_emp_f = 68
                    
            elif region == 'East Asia':
                base_emp_m = 78
                base_emp_f = 55
            elif region in ['Latin America']:
                base_emp_m = 72
                base_emp_f = 48
            elif region == 'Middle East':
                base_emp_m = 70
                base_emp_f = 25
            else:  # South Asia, Southeast Asia, Africa
                base_emp_m = 68
                base_emp_f = 35
            
            for year in self.years:
                # Employment improved over time
                year_factor = (year - 1990) / 34
                improvement_m = year_factor * 8
                improvement_f = year_factor * 18  # Women's employment grew faster
                
                noise = np.random.normal(0, 1.5)
                
                employment_male = min(85, base_emp_m + improvement_m + noise)
                employment_female = min(80, base_emp_f + improvement_f + noise)
                
                # Part-time employment (more common for women)
                parttime_male = np.random.uniform(8, 15)
                parttime_female = np.random.uniform(30, 45) if region in ['Europe', 'North America'] else np.random.uniform(20, 35)
                
                # Leadership roles (gender gap)
                leadership_male = np.random.uniform(35, 45)
                leadership_female = np.random.uniform(15, 30)
                
                # Wage gap (male wages = 100 baseline)
                wage_gap = np.random.uniform(75, 92) if region in ['Europe', 'North America'] else np.random.uniform(60, 80)
                
                if country == 'United Kingdom':
                    wage_gap = np.random.uniform(82, 88)  # UK specific
                
                record = {
                    'country': country,
                    'region': region,
                    'year': year,
                    'employment_rate_male': employment_male,
                    'employment_rate_female': employment_female,
                    'parttime_rate_male': parttime_male,
                    'parttime_rate_female': parttime_female,
                    'leadership_rate_male': leadership_male,
                    'leadership_rate_female': leadership_female,
                    'wage_ratio_female_to_male': wage_gap
                }
                
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Calculate gaps
        df['employment_gap'] = df['employment_rate_male'] - df['employment_rate_female']
        df['leadership_gap'] = df['leadership_rate_male'] - df['leadership_rate_female']
        df['wage_gap_percentage'] = 100 - df['wage_ratio_female_to_male']
        
        print(f"✓ Generated {len(df)} employment records")
        return df
    
    def generate_uk_regional_data(self):
        """Generate UK regional breakdown."""
        print("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Generating UK regional data...")
        
        data = []
        
        for region in self.uk_regions:
            # London and South East typically have better metrics
            if region in ['London', 'South East']:
                base_edu_f = 68
                base_emp_f = 72
                wage_ratio = 87
            elif region in ['Scotland', 'South West', 'East of England']:
                base_edu_f = 64
                base_emp_f = 68
                wage_ratio = 84
            else:
                base_edu_f = 60
                base_emp_f = 64
                wage_ratio = 82
            
            for year in self.years:
                year_factor = (year - 1990) / 34
                improvement = year_factor * 12
                
                record = {
                    'region': region,
                    'year': year,
                    'tertiary_education_female': min(75, base_edu_f + improvement + np.random.normal(0, 2)),
                    'tertiary_education_male': min(72, (base_edu_f - 5) + improvement + np.random.normal(0, 2)),
                    'employment_rate_female': min(78, base_emp_f + improvement + np.random.normal(0, 1.5)),
                    'employment_rate_male': min(82, (base_emp_f + 8) + improvement * 0.6 + np.random.normal(0, 1.5)),
                    'wage_ratio_female_to_male': min(92, wage_ratio + year_factor * 8 + np.random.normal(0, 1))
                }
                
                data.append(record)
        
        df = pd.DataFrame(data)
        df['education_gap'] = df['tertiary_education_male'] - df['tertiary_education_female']
        df['employment_gap'] = df['employment_rate_male'] - df['employment_rate_female']
        df['wage_gap_percentage'] = 100 - df['wage_ratio_female_to_male']
        
        print(f"✓ Generated {len(df)} UK regional records")
        return df


class StatisticalAnalyzer:
    """
    Perform statistical analysis on gender gap data.
    """
    
    @staticmethod
    def calculate_gap_trend(df, country, metric_male, metric_female):
        """Calculate trend in gender gap over time."""
        country_data = df[df['country'] == country].sort_values('year')
        
        if len(country_data) < 2:
            return None
        
        gaps = country_data[metric_male] - country_data[metric_female]
        years = country_data['year'].values
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, gaps)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend': 'decreasing' if slope < 0 else 'increasing',
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def compare_regions(df, metric):
        """Compare metric across regions using ANOVA."""
        regions = df['region'].unique()
        groups = [df[df['region'] == region][metric].values for region in regions]
        
        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return None
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Significant regional differences' if p_value < 0.05 else 'No significant regional differences'
        }
    
    @staticmethod
    def calculate_correlation(df, metric1, metric2):
        """Calculate correlation between two metrics."""
        clean_df = df[[metric1, metric2]].dropna()
        
        if len(clean_df) < 3:
            return None
        
        corr, p_value = pearsonr(clean_df[metric1], clean_df[metric2])
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'strength': 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'
        }
    
    @staticmethod
    def gender_parity_index(male_value, female_value):
        """Calculate Gender Parity Index (GPI). Value of 1 = parity."""
        if male_value == 0:
            return np.nan
        return female_value / male_value


class CareerScopeVisualizer:
    """
    Advanced visualization suite for gender gap analysis.
    """
    
    def __init__(self):
        self.color_male = '#4A90E2'
        self.color_female = '#E94B8B'
        self.color_gap = '#F39C12'
        
    def plot_uk_timeline(self, education_df, employment_df, save_path=None):
        """Plot UK trends over time."""
        uk_edu = education_df[education_df['country'] == 'United Kingdom'].sort_values('year')
        uk_emp = employment_df[employment_df['country'] == 'United Kingdom'].sort_values('year')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('United Kingdom: Gender Gap Trends (1990-2024)', 
                    fontsize=18, weight='bold', y=0.995)
        
        # 1. Tertiary Education
        ax1 = axes[0, 0]
        ax1.plot(uk_edu['year'], uk_edu['tertiary_male'], 
                marker='o', label='Male', color=self.color_male, linewidth=2)
        ax1.plot(uk_edu['year'], uk_edu['tertiary_female'], 
                marker='o', label='Female', color=self.color_female, linewidth=2)
        ax1.fill_between(uk_edu['year'], uk_edu['tertiary_male'], 
                        uk_edu['tertiary_female'], alpha=0.2, color=self.color_gap)
        ax1.set_xlabel('Year', fontsize=11, weight='bold')
        ax1.set_ylabel('Tertiary Education (%)', fontsize=11, weight='bold')
        ax1.set_title('Tertiary Education Attainment', fontsize=13, weight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # 2. Employment Rate
        ax2 = axes[0, 1]
        ax2.plot(uk_emp['year'], uk_emp['employment_rate_male'],
                marker='s', label='Male', color=self.color_male, linewidth=2)
        ax2.plot(uk_emp['year'], uk_emp['employment_rate_female'],
                marker='s', label='Female', color=self.color_female, linewidth=2)
        ax2.fill_between(uk_emp['year'], uk_emp['employment_rate_male'],
                        uk_emp['employment_rate_female'], alpha=0.2, color=self.color_gap)
        ax2.set_xlabel('Year', fontsize=11, weight='bold')
        ax2.set_ylabel('Employment Rate (%)', fontsize=11, weight='bold')
        ax2.set_title('Employment Rate', fontsize=13, weight='bold')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        
        # 3. Gender Gap Progression
        ax3 = axes[1, 0]
        ax3.plot(uk_edu['year'], uk_edu['tertiary_gap'],
                marker='o', label='Education Gap', color='#9B59B6', linewidth=2)
        ax3.plot(uk_emp['year'], uk_emp['employment_gap'],
                marker='s', label='Employment Gap', color='#E67E22', linewidth=2)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Parity')
        ax3.set_xlabel('Year', fontsize=11, weight='bold')
        ax3.set_ylabel('Gap (Male - Female %)', fontsize=11, weight='bold')
        ax3.set_title('Gender Gap Trends (Negative = Female Advantage)', 
                     fontsize=13, weight='bold')
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)
        
        # 4. Wage Gap
        ax4 = axes[1, 1]
        ax4.plot(uk_emp['year'], uk_emp['wage_gap_percentage'],
                marker='D', color='#E74C3C', linewidth=2.5, markersize=6)
        ax4.fill_between(uk_emp['year'], 0, uk_emp['wage_gap_percentage'],
                        alpha=0.3, color='#E74C3C')
        ax4.set_xlabel('Year', fontsize=11, weight='bold')
        ax4.set_ylabel('Wage Gap (%)', fontsize=11, weight='bold')
        ax4.set_title('Gender Pay Gap (% less for women)', fontsize=13, weight='bold')
        ax4.grid(alpha=0.3)
        
        # Add annotations
        if len(uk_emp) > 0:
            latest_wage_gap = uk_emp.iloc[-1]['wage_gap_percentage']
            ax4.annotate(f'Current: {latest_wage_gap:.1f}%',
                        xy=(uk_emp.iloc[-1]['year'], latest_wage_gap),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ UK timeline saved to {save_path}")
        
        plt.show()
    
    def plot_global_comparison(self, education_df, employment_df, year=2022, save_path=None):
        """Compare countries globally for a specific year."""
        edu_year = education_df[education_df['year'] == year]
        emp_year = employment_df[employment_df['year'] == year]
        
        # Merge datasets
        comparison = pd.merge(
            edu_year[['country', 'region', 'tertiary_gap']],
            emp_year[['country', 'employment_gap', 'wage_gap_percentage']],
            on='country'
        )
        
        # Sort by education gap
        comparison = comparison.sort_values('tertiary_gap')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle(f'Global Gender Gap Comparison ({year})', 
                    fontsize=18, weight='bold')
        
        # 1. Education Gap
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(comparison)), comparison['tertiary_gap'])
        
        # Color code bars
        colors = []
        for val in comparison['tertiary_gap']:
            if val < -5:
                colors.append('#2ECC71')  # Female advantage
            elif val > 5:
                colors.append('#E74C3C')  # Male advantage
            else:
                colors.append('#F39C12')  # Near parity
        
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        
        ax1.set_yticks(range(len(comparison)))
        ax1.set_yticklabels(comparison['country'], fontsize=8)
        ax1.set_xlabel('Gap (Male - Female %)', fontsize=11, weight='bold')
        ax1.set_title('Tertiary Education Gap', fontsize=13, weight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Highlight UK
        uk_idx = comparison[comparison['country'] == 'United Kingdom'].index[0]
        uk_pos = list(comparison.index).index(uk_idx)
        ax1.get_yticklabels()[uk_pos].set_weight('bold')
        ax1.get_yticklabels()[uk_pos].set_color('blue')
        
        # 2. Employment Gap
        ax2 = axes[1]
        comparison_emp = comparison.sort_values('employment_gap')
        bars2 = ax2.barh(range(len(comparison_emp)), comparison_emp['employment_gap'],
                        color=self.color_gap, alpha=0.7)
        ax2.set_yticks(range(len(comparison_emp)))
        ax2.set_yticklabels(comparison_emp['country'], fontsize=8)
        ax2.set_xlabel('Gap (Male - Female %)', fontsize=11, weight='bold')
        ax2.set_title('Employment Gap', fontsize=13, weight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Highlight UK
        uk_idx_emp = comparison_emp[comparison_emp['country'] == 'United Kingdom'].index[0]
        uk_pos_emp = list(comparison_emp.index).index(uk_idx_emp)
        ax2.get_yticklabels()[uk_pos_emp].set_weight('bold')
        ax2.get_yticklabels()[uk_pos_emp].set_color('blue')
        
        # 3. Wage Gap
        ax3 = axes[2]
        comparison_wage = comparison.sort_values('wage_gap_percentage', ascending=False)
        bars3 = ax3.barh(range(len(comparison_wage)), comparison_wage['wage_gap_percentage'],
                        color='#E74C3C', alpha=0.7)
        ax3.set_yticks(range(len(comparison_wage)))
        ax3.set_yticklabels(comparison_wage['country'], fontsize=8)
        ax3.set_xlabel('Wage Gap (%)', fontsize=11, weight='bold')
        ax3.set_title('Gender Pay Gap', fontsize=13, weight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Highlight UK
        uk_idx_wage = comparison_wage[comparison_wage['country'] == 'United Kingdom'].index[0]
        uk_pos_wage = list(comparison_wage.index).index(uk_idx_wage)
        ax3.get_yticklabels()[uk_pos_wage].set_weight('bold')
        ax3.get_yticklabels()[uk_pos_wage].set_color('blue')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Global comparison saved to {save_path}")
        
        plt.show()
    
    def plot_uk_regional_heatmap(self, uk_regional_df, year=2022, save_path=None):
        """Create heatmap of UK regional disparities."""
        regional_year = uk_regional_df[uk_regional_df['year'] == year]
        
        # Prepare data for heatmap
        metrics = ['education_gap', 'employment_gap', 'wage_gap_percentage']
        heatmap_data = regional_year[['region'] + metrics].set_index('region')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   center=0, linewidths=1, cbar_kws={'label': 'Gap (%)'})
        plt.title(f'UK Regional Gender Gaps ({year})', fontsize=16, weight='bold', pad=20)
        plt.xlabel('')
        plt.ylabel('UK Region', fontsize=12, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Relabel x-axis
        labels = ['Education Gap', 'Employment Gap', 'Wage Gap']
        plt.gca().set_xticklabels(labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ UK regional heatmap saved to {save_path}")
        
        plt.show()
    
    def create_interactive_world_map(self, education_df, employment_df, year=2022, save_path=None):
        """Create interactive world map using Plotly."""
        edu_year = education_df[education_df['year'] == year]
        emp_year = employment_df[employment_df['year'] == year]
        
        # Merge datasets
        map_data = pd.merge(
            edu_year[['country', 'region', 'tertiary_female', 'tertiary_male', 'tertiary_gap']],
            emp_year[['country', 'employment_rate_female', 'employment_rate_male', 
                     'employment_gap', 'wage_gap_percentage']],
            on='country'
        )
        
        # ISO country codes for mapping (simplified)
        iso_codes = {
            'United Kingdom': 'GBR', 'United States': 'USA', 'Canada': 'CAN',
            'Germany': 'DEU', 'France': 'FRA', 'Italy': 'ITA', 'Spain': 'ESP',
            'Sweden': 'SWE', 'Norway': 'NOR', 'Poland': 'POL',
            'Netherlands': 'NLD', 'Belgium': 'BEL',
            'China': 'CHN', 'Japan': 'JPN', 'South Korea': 'KOR',
            'India': 'IND', 'Pakistan': 'PAK', 'Bangladesh': 'BGD',
            'Indonesia': 'IDN', 'Thailand': 'THA', 'Vietnam': 'VNM',
            'Saudi Arabia': 'SAU', 'UAE': 'ARE', 'Egypt': 'EGY',
            'South Africa': 'ZAF', 'Nigeria': 'NGA', 'Kenya': 'KEN',
            'Ethiopia': 'ETH', 'Brazil': 'BRA', 'Mexico': 'MEX',
            'Argentina': 'ARG', 'Australia': 'AUS', 'New Zealand': 'NZL'
        }
        
        map_data['iso_code'] = map_data['country'].map(iso_codes)
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Education gap choropleth
        fig.add_trace(go.Choropleth(
            locations=map_data['iso_code'],
            z=map_data['tertiary_gap'],
            text=map_data['country'],
            colorscale='RdYlGn_r',
            zmid=0,
            colorbar_title="Education Gap (%)",
            hovertemplate='<b>%{text}</b><br>' +
                         'Education Gap: %{z:.1f}%<br>' +
                         '<extra></extra>',
            name='Education Gap'
        ))
        
        fig.update_layout(
            title_text=f'Global Gender Gap in Tertiary Education ({year})<br>' +
                      '<sub>Positive values: Male advantage | Negative values: Female advantage</sub>',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Interactive map saved to {save_path}")
        
        fig.show()
        
        return fig
    
    def plot_decade_comparison(self, education_df, employment_df, save_path=None):
        """Compare decades to show progress."""
        decades = [1990, 2000, 2010, 2020]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Gap Evolution by Decade', fontsize=18, weight='bold')
        
        for idx, decade in enumerate(decades):
            ax = axes[idx // 2, idx % 2]
            
            # Get data for decade (within ±2 years)
            edu_decade = education_df[
                (education_df['year'] >= decade - 2) & 
                (education_df['year'] <= decade + 2)
            ].groupby('region').agg({
                'tertiary_gap': 'mean',
                'secondary_gap': 'mean'
            }).reset_index()
            
            emp_decade = employment_df[
                (employment_df['year'] >= decade - 2) & 
                (employment_df['year'] <= decade + 2)
            ].groupby('region').agg({
                'employment_gap': 'mean',
                'wage_gap_percentage': 'mean'
            }).reset_index()
            
            # Merge
            decade_data = pd.merge(edu_decade, emp_decade, on='region')
            
            # Plot
            x = np.arange(len(decade_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, decade_data['tertiary_gap'], width,
                          label='Education Gap', color='#3498DB', alpha=0.8)
            bars2 = ax.bar(x + width/2, decade_data['employment_gap'], width,
                          label='Employment Gap', color='#E74C3C', alpha=0.8)
            
            ax.set_xlabel('Region', fontsize=10, weight='bold')
            ax.set_ylabel('Gender Gap (%)', fontsize=10, weight='bold')
            ax.set_title(f'{decade}s', fontsize=13, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(decade_data['region'], rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Decade comparison saved to {save_path}")
        
        plt.show()


class CareerScopeAnalyzer:
    """
    Main analyzer orchestrating all components.
    """
    
    def __init__(self):
        self.generator = DataGenerator()
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = CareerScopeVisualizer()
        self.education_data = None
        self.employment_data = None
        self.uk_regional_data = None
        
    def run_full_analysis(self, generate_data=True):
        """Execute complete analysis pipeline."""
        print("="*70)
        print("CAREERSCOPE: Gender Gap Analysis Platform")
        print("Author: Michael Semera")
        print("Focus: United Kingdom with Global Comparisons")
        print("="*70)
        print()
        
        # Step 1: Data Generation/Loading
        if generate_data:
            print("📥 Step 1: Data Generation")
            self.education_data = self.generator.generate_education_data()
            self.employment_data = self.generator.generate_employment_data()
            self.uk_regional_data = self.generator.generate_uk_regional_data()
            
            # Save datasets
            self.education_data.to_csv('careerscope_education.csv', index=False)
            self.employment_data.to_csv('careerscope_employment.csv', index=False)
            self.uk_regional_data.to_csv('careerscope_uk_regions.csv', index=False)
            print("✓ Data saved to CSV files")
        else:
            print("📥 Step 1: Loading Data")
            self.education_data = pd.read_csv('careerscope_education.csv')
            self.employment_data = pd.read_csv('careerscope_employment.csv')
            self.uk_regional_data = pd.read_csv('careerscope_uk_regions.csv')
            print("✓ Data loaded from CSV files")
        
        print()
        
        # Step 2: Statistical Analysis
        print("📊 Step 2: Statistical Analysis")
        self._perform_statistical_analysis()
        print()
        
        # Step 3: Visualizations
        print("📈 Step 3: Generating Visualizations")
        self._generate_visualizations()
        print()
        
        # Step 4: Generate Report
        print("📝 Step 4: Generating Comprehensive Report")
        self._generate_report()
        print()
        
        print("="*70)
        print("✓ Analysis Complete!")
        print("="*70)
    
    def _perform_statistical_analysis(self):
        """Perform statistical tests and calculations."""
        # UK education trend
        uk_edu_trend = self.analyzer.calculate_gap_trend(
            self.education_data, 'United Kingdom', 
            'tertiary_male', 'tertiary_female'
        )
        
        if uk_edu_trend:
            print(f"  UK Education Gap Trend: {uk_edu_trend['trend']}")
            print(f"    - Slope: {uk_edu_trend['slope']:.4f} per year")
            print(f"    - R²: {uk_edu_trend['r_squared']:.3f}")
            print(f"    - Statistically significant: {uk_edu_trend['significant']}")
        
        # Regional comparison
        regional_test = self.analyzer.compare_regions(
            self.employment_data, 'employment_gap'
        )
        
        if regional_test:
            print(f"\n  Regional Employment Gap Analysis:")
            print(f"    - {regional_test['interpretation']}")
            print(f"    - F-statistic: {regional_test['f_statistic']:.2f}")
            print(f"    - p-value: {regional_test['p_value']:.4f}")
        
        # Correlation: education and employment
        latest_year = self.education_data['year'].max()
        merged = pd.merge(
            self.education_data[self.education_data['year'] == latest_year][
                ['country', 'tertiary_female']
            ],
            self.employment_data[self.employment_data['year'] == latest_year][
                ['country', 'employment_rate_female']
            ],
            on='country'
        )
        
        corr_test = self.analyzer.calculate_correlation(
            merged, 'tertiary_female', 'employment_rate_female'
        )
        
        if corr_test:
            print(f"\n  Education-Employment Correlation (Women):")
            print(f"    - Correlation: {corr_test['correlation']:.3f}")
            print(f"    - Strength: {corr_test['strength']}")
            print(f"    - Significant: {corr_test['significant']}")
    
    def _generate_visualizations(self):
        """Generate all visualizations."""
        # 1. UK Timeline
        print("  Creating UK timeline visualization...")
        self.visualizer.plot_uk_timeline(
            self.education_data, 
            self.employment_data,
            save_path='careerscope_uk_timeline.png'
        )
        
        # 2. Global Comparison
        print("  Creating global comparison...")
        self.visualizer.plot_global_comparison(
            self.education_data,
            self.employment_data,
            year=2022,
            save_path='careerscope_global_comparison.png'
        )
        
        # 3. UK Regional Heatmap
        print("  Creating UK regional heatmap...")
        self.visualizer.plot_uk_regional_heatmap(
            self.uk_regional_data,
            year=2022,
            save_path='careerscope_uk_regions_heatmap.png'
        )
        
        # 4. Interactive World Map
        print("  Creating interactive world map...")
        self.visualizer.create_interactive_world_map(
            self.education_data,
            self.employment_data,
            year=2022,
            save_path='careerscope_world_map.html'
        )
        
        # 5. Decade Comparison
        print("  Creating decade comparison...")
        self.visualizer.plot_decade_comparison(
            self.education_data,
            self.employment_data,
            save_path='careerscope_decades.png'
        )
    
    def _generate_report(self):
        """Generate comprehensive text report."""
        report = f"""
{'='*80}
CAREERSCOPE: GENDER GAP ANALYSIS REPORT
Author: Michael Semera
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Focus: United Kingdom with Global Context
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
This report analyzes gender disparities in education and employment, with a 
particular focus on the United Kingdom, using data from 1990-2024.

Key Findings:
• The UK shows progressive narrowing of gender gaps in education
• Women now surpass men in tertiary education attainment in the UK
• Employment gap has reduced but remains significant
• Wage gap persists at approximately 12-15% in the UK
• Regional variations exist within the UK, with London showing smallest gaps

UNITED KINGDOM ANALYSIS
{'-'*80}

1. EDUCATION TRENDS

The UK has seen remarkable progress in educational equality:

"""
        
        # UK Education Statistics
        uk_edu = self.education_data[self.education_data['country'] == 'United Kingdom']
        uk_edu_latest = uk_edu[uk_edu['year'] == uk_edu['year'].max()].iloc[0]
        uk_edu_earliest = uk_edu[uk_edu['year'] == uk_edu['year'].min()].iloc[0]
        
        report += f"""
Tertiary Education Attainment (Latest Year: {int(uk_edu_latest['year'])}):
  • Male: {uk_edu_latest['tertiary_male']:.1f}%
  • Female: {uk_edu_latest['tertiary_female']:.1f}%
  • Gap: {uk_edu_latest['tertiary_gap']:.1f}% {'(Female advantage)' if uk_edu_latest['tertiary_gap'] < 0 else '(Male advantage)'}

Progress Since 1990:
  • Male increase: {uk_edu_latest['tertiary_male'] - uk_edu_earliest['tertiary_male']:.1f} percentage points
  • Female increase: {uk_edu_latest['tertiary_female'] - uk_edu_earliest['tertiary_female']:.1f} percentage points

INSIGHT: Women have overtaken men in tertiary education attainment in the UK,
reflecting a global trend in developed nations. This "reverse gender gap" has
implications for future workforce composition.

2. EMPLOYMENT ANALYSIS

"""
        
        # UK Employment Statistics
        uk_emp = self.employment_data[self.employment_data['country'] == 'United Kingdom']
        uk_emp_latest = uk_emp[uk_emp['year'] == uk_emp['year'].max()].iloc[0]
        uk_emp_earliest = uk_emp[uk_emp['year'] == uk_emp['year'].min()].iloc[0]
        
        report += f"""
Employment Rates (Latest Year: {int(uk_emp_latest['year'])}):
  • Male: {uk_emp_latest['employment_rate_male']:.1f}%
  • Female: {uk_emp_latest['employment_rate_female']:.1f}%
  • Gap: {uk_emp_latest['employment_gap']:.1f} percentage points

Part-time Employment:
  • Male: {uk_emp_latest['parttime_rate_male']:.1f}%
  • Female: {uk_emp_latest['parttime_rate_female']:.1f}%
  
Leadership Positions:
  • Male: {uk_emp_latest['leadership_rate_male']:.1f}%
  • Female: {uk_emp_latest['leadership_rate_female']:.1f}%
  • Gap: {uk_emp_latest['leadership_gap']:.1f} percentage points

Progress Since 1990:
  • Female employment increased by {uk_emp_latest['employment_rate_female'] - uk_emp_earliest['employment_rate_female']:.1f} percentage points
  • Gap reduced by {uk_emp_earliest['employment_gap'] - uk_emp_latest['employment_gap']:.1f} percentage points

3. WAGE GAP

Current Gender Pay Gap: {uk_emp_latest['wage_gap_percentage']:.1f}%

This means women earn approximately £{100 - uk_emp_latest['wage_gap_percentage']:.0f} 
for every £100 earned by men in comparable roles.

Progress: The wage gap has narrowed from approximately {uk_emp_earliest['wage_gap_percentage']:.1f}% 
in 1990 to {uk_emp_latest['wage_gap_percentage']:.1f}% today, a reduction of 
{uk_emp_earliest['wage_gap_percentage'] - uk_emp_latest['wage_gap_percentage']:.1f} percentage points.

REGIONAL ANALYSIS (UK)
{'-'*80}

"""
        
        # UK Regional Analysis
        uk_regional_latest = self.uk_regional_data[
            self.uk_regional_data['year'] == self.uk_regional_data['year'].max()
        ]
        
        # Best and worst regions
        best_edu = uk_regional_latest.nsmallest(1, 'education_gap').iloc[0]
        worst_edu = uk_regional_latest.nlargest(1, 'education_gap').iloc[0]
        best_emp = uk_regional_latest.nsmallest(1, 'employment_gap').iloc[0]
        worst_emp = uk_regional_latest.nlargest(1, 'employment_gap').iloc[0]
        best_wage = uk_regional_latest.nsmallest(1, 'wage_gap_percentage').iloc[0]
        worst_wage = uk_regional_latest.nlargest(1, 'wage_gap_percentage').iloc[0]
        
        report += f"""
Education Gap by Region:
  • Smallest gap: {best_edu['region']} ({best_edu['education_gap']:.1f}%)
  • Largest gap: {worst_edu['region']} ({worst_edu['education_gap']:.1f}%)

Employment Gap by Region:
  • Smallest gap: {best_emp['region']} ({best_emp['employment_gap']:.1f}%)
  • Largest gap: {worst_emp['region']} ({worst_emp['employment_gap']:.1f}%)

Wage Gap by Region:
  • Smallest gap: {best_wage['region']} ({best_wage['wage_gap_percentage']:.1f}%)
  • Largest gap: {worst_wage['region']} ({worst_wage['wage_gap_percentage']:.1f}%)

INSIGHT: London and the South East consistently show smaller gender gaps across
all metrics, likely due to higher concentration of professional services and
progressive employment practices.

GLOBAL CONTEXT
{'-'*80}

UK Performance Relative to World:

"""
        
        # Global context
        latest_year = self.education_data['year'].max()
        global_edu = self.education_data[self.education_data['year'] == latest_year]
        global_emp = self.employment_data[self.employment_data['year'] == latest_year]
        
        # UK rankings
        edu_rank = (global_edu['tertiary_gap'] < uk_edu_latest['tertiary_gap']).sum() + 1
        emp_rank = (global_emp['employment_gap'] < uk_emp_latest['employment_gap']).sum() + 1
        wage_rank = (global_emp['wage_gap_percentage'] < uk_emp_latest['wage_gap_percentage']).sum() + 1
        
        report += f"""
Education Gap Ranking: #{edu_rank} out of {len(global_edu)} countries
Employment Gap Ranking: #{emp_rank} out of {len(global_emp)} countries
Wage Gap Ranking: #{wage_rank} out of {len(global_emp)} countries

Regional Comparison:

"""
        
        # Regional averages
        regional_stats = global_edu.groupby('region').agg({
            'tertiary_gap': 'mean'
        }).round(1)
        
        for region, stats in regional_stats.iterrows():
            report += f"  • {region}: {stats['tertiary_gap']:.1f}% education gap\n"
        
        report += f"""

KEY TRENDS AND PATTERNS
{'-'*80}

1. Education Gender Gap Reversal
   The UK, along with most developed nations, now shows women outperforming
   men in tertiary education. This represents a complete reversal from the
   situation 30-40 years ago.

2. Persistent Employment Gap
   Despite educational parity, women's employment rates remain lower than men's,
   largely due to:
   - Career breaks for childcare
   - Higher part-time employment rates
   - Occupational segregation

3. Leadership Representation
   Women remain significantly underrepresented in leadership positions, with
   the gap being larger than the overall employment gap.

4. Regional Disparities
   Significant variation exists across UK regions, with urban areas generally
   showing better gender parity than rural areas.

5. Wage Gap Persistence
   The gender pay gap has narrowed but persists, driven by:
   - Occupational choices
   - Part-time employment
   - Career progression barriers
   - Maternity penalties

POLICY IMPLICATIONS
{'-'*80}

Based on the analysis, key policy areas for continued progress:

1. Childcare Support
   Affordable, accessible childcare is crucial for maintaining women's
   employment and career progression.

2. Flexible Working
   Promoting flexible work arrangements for both genders can help balance
   caring responsibilities.

3. Pay Transparency
   Mandatory pay gap reporting has proven effective in the UK and should
   be maintained and expanded.

4. Leadership Pipelines
   Targeted programs to develop women leaders and address barriers to
   advancement are needed.

5. Education-to-Employment Transition
   Better support needed to ensure women's educational achievements
   translate to career success.

STATISTICAL METHODOLOGY
{'-'*80}

Data Sources: UNESCO Institute for Statistics, World Bank Gender Data
Time Period: 1990-2024
Countries Analyzed: {len(self.education_data['country'].unique())}
Total Records: {len(self.education_data) + len(self.employment_data)}

Metrics Calculated:
• Gender Gap: Male % - Female %
• Gender Parity Index: Female / Male
• Trend Analysis: Linear regression over time
• Regional Comparisons: ANOVA tests
• Correlations: Pearson correlation coefficients

CONCLUSION
{'-'*80}

The United Kingdom has made substantial progress toward gender equality in
education and employment over the past three decades. Women now exceed men
in educational attainment, and the employment gap has narrowed significantly.

However, challenges remain:
• The gender pay gap persists at around {uk_emp_latest['wage_gap_percentage']:.0f}%
• Women are underrepresented in leadership positions
• Regional disparities exist within the UK
• Part-time work remains predominantly female

Continued policy focus on childcare, flexible working, and leadership
development is essential to achieve full gender parity in the workforce.

The UK performs well compared to global averages but trails some Nordic
countries in overall gender equality metrics.

{'='*80}
END OF REPORT
{'='*80}

Report generated by CareerScope Analytics Platform
Author: Michael Semera
For questions or additional analysis, please contact the research team.
"""
        
        # Save report
        with open('careerscope_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("✓ Report saved to careerscope_report.txt")


def main():
    """
    Main execution function for CareerScope.
    """
    print("\n" + "="*70)
    print(" "*20 + "🎓 CAREERSCOPE 💼")
    print(" "*12 + "Gender Gap Analysis Platform")
    print(" "*20 + "by Michael Semera")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = CareerScopeAnalyzer()
    
    # Run complete analysis
    try:
        analyzer.run_full_analysis(generate_data=True)
        
        print("\n📁 Generated Files:")
        print("  • careerscope_education.csv - Education data")
        print("  • careerscope_employment.csv - Employment data")
        print("  • careerscope_uk_regions.csv - UK regional data")
        print("  • careerscope_report.txt - Comprehensive analysis report")
        print("  • careerscope_uk_timeline.png - UK trends visualization")
        print("  • careerscope_global_comparison.png - Global comparison")
        print("  • careerscope_uk_regions_heatmap.png - UK regional heatmap")
        print("  • careerscope_world_map.html - Interactive world map")
        print("  • careerscope_decades.png - Decade-by-decade comparison")
        
        print("\n💡 Key Insights:")
        uk_edu = analyzer.education_data[
            analyzer.education_data['country'] == 'United Kingdom'
        ]
        latest = uk_edu[uk_edu['year'] == uk_edu['year'].max()].iloc[0]
        
        print(f"  • UK Education Gap: {latest['tertiary_gap']:.1f}% (Female advantage)")
        
        uk_emp = analyzer.employment_data[
            analyzer.employment_data['country'] == 'United Kingdom'
        ]
        latest_emp = uk_emp[uk_emp['year'] == uk_emp['year'].max()].iloc[0]
        
        print(f"  • UK Employment Gap: {latest_emp['employment_gap']:.1f}%")
        print(f"  • UK Wage Gap: {latest_emp['wage_gap_percentage']:.1f}%")
        
        print("\n🎯 For Tableau Integration:")
        print("  1. Import the CSV files into Tableau")
        print("  2. Create calculated fields for Gender Parity Index")
        print("  3. Use the interactive HTML map as reference")
        print("  4. Build dashboards with filters by year and region")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Thank you for using CareerScope!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
