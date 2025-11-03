"""
Advanced Analysis & Visualization Module for Privacy-LBS
Provides detailed comparative analysis, attack simulations, and advanced visualizations

This module includes:
- Privacy attack simulations
- Comparative technique analysis
- Advanced visualization dashboards
- Performance benchmarking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Set visualization style
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# ATTACK SIMULATION
# ============================================================================

class PrivacyAttackSimulator:
    """Simulates various privacy attacks on anonymized data"""
    
    def __init__(self):
        self.attack_results = {}
    
    def simulate_linkage_attack(self, original_df: pd.DataFrame, 
                                anonymized_df: pd.DataFrame) -> Dict:
        """Simulate identity linkage attack"""
        print("\n🔴 Simulating Linkage Attack...")
        
        # Try to match anonymized locations to original based on patterns
        matches = 0
        total = min(len(original_df), len(anonymized_df))
        
        for i in range(min(100, total)):  # Sample for performance
            orig = original_df.iloc[i]
            anon = anonymized_df.iloc[i]
            
            # Calculate distance
            dist = self._haversine_distance(
                orig['latitude'], orig['longitude'],
                anon['latitude'], anon['longitude']
            )
            
            # If very close, consider it a successful re-identification
            if dist < 100:  # Within 100m
                matches += 1
        
        success_rate = (matches / min(100, total)) * 100
        
        result = {
            "attack_type": "linkage",
            "success_rate": success_rate,
            "samples_tested": min(100, total),
            "successful_matches": matches
        }
        
        print(f"  Attack Success Rate: {success_rate:.2f}%")
        print(f"  Successful Matches: {matches}/{min(100, total)}")
        
        self.attack_results['linkage'] = result
        return result
    
    def simulate_trajectory_attack(self, location_data: pd.DataFrame) -> Dict:
        """Simulate trajectory tracking attack"""
        print("\n🔴 Simulating Trajectory Tracking Attack...")
        
        # Group by user and check if trajectories can be reconstructed
        trackable_users = 0
        total_users = location_data['user_id'].nunique()
        
        for user_id in location_data['user_id'].unique()[:50]:  # Sample
            user_data = location_data[location_data['user_id'] == user_id]
            
            # If we can see a clear movement pattern, trajectory is trackable
            if len(user_data) > 5:
                locations = user_data[['latitude', 'longitude']].values
                
                # Check for consistent movement pattern
                distances = []
                for i in range(len(locations) - 1):
                    dist = self._haversine_distance(
                        locations[i][0], locations[i][1],
                        locations[i+1][0], locations[i+1][1]
                    )
                    distances.append(dist)
                
                # If average distance is small, trajectory is traceable
                if np.mean(distances) < 5000:  # Within 5km
                    trackable_users += 1
        
        success_rate = (trackable_users / min(50, total_users)) * 100
        
        result = {
            "attack_type": "trajectory",
            "success_rate": success_rate,
            "trackable_users": trackable_users,
            "total_tested": min(50, total_users)
        }
        
        print(f"  Attack Success Rate: {success_rate:.2f}%")
        print(f"  Trackable Users: {trackable_users}/{min(50, total_users)}")
        
        self.attack_results['trajectory'] = result
        return result
    
    def simulate_inference_attack(self, location_data: pd.DataFrame) -> Dict:
        """Simulate sensitive attribute inference attack"""
        print("\n🔴 Simulating Inference Attack...")
        
        # Try to infer sensitive attributes from visited locations
        # e.g., religion from places of worship, health from hospitals
        
        sensitive_locations = {
            "religious": [(21.15, 79.09)],  # Mock religious site
            "medical": [(21.14, 79.08)],     # Mock hospital
            "political": [(21.16, 79.10)]    # Mock political office
        }
        
        inferred_attributes = 0
        users_checked = 0
        
        for user_id in location_data['user_id'].unique()[:50]:
            users_checked += 1
            user_data = location_data[location_data['user_id'] == user_id]
            
            for _, row in user_data.iterrows():
                for category, locations in sensitive_locations.items():
                    for sens_lat, sens_lon in locations:
                        dist = self._haversine_distance(
                            row['latitude'], row['longitude'],
                            sens_lat, sens_lon
                        )
                        if dist < 500:  # Within 500m
                            inferred_attributes += 1
                            break
        
        success_rate = (inferred_attributes / (users_checked * len(sensitive_locations))) * 100
        
        result = {
            "attack_type": "inference",
            "success_rate": min(success_rate, 100),
            "attributes_inferred": inferred_attributes,
            "users_tested": users_checked
        }
        
        print(f"  Attack Success Rate: {min(success_rate, 100):.2f}%")
        print(f"  Attributes Inferred: {inferred_attributes}")
        
        self.attack_results['inference'] = result
        return result
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance in meters"""
        R = 6371000
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def generate_attack_report(self) -> str:
        """Generate attack simulation report"""
        report = f"""
{'='*80}
PRIVACY ATTACK SIMULATION REPORT
{'='*80}

"""
        for attack_type, results in self.attack_results.items():
            report += f"\n{attack_type.upper()} ATTACK:\n"
            for key, value in results.items():
                report += f"  {key}: {value}\n"
        
        report += f"\n{'='*80}\n"
        return report

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

class ComparativeAnalyzer:
    """Performs comparative analysis across techniques"""
    
    def __init__(self):
        self.comparison_data = {}
    
    def compare_techniques(self, results_dict: Dict) -> pd.DataFrame:
        """Compare all techniques across multiple dimensions"""
        print("\n📊 Performing Comparative Analysis...")
        
        comparison = []
        
        for technique, metrics in results_dict.items():
            comparison.append({
                'Technique': technique,
                'Re-ID Risk (%)': metrics.get('reidentification_risk', 0),
                'Info Loss (m)': metrics.get('information_loss', 0),
                'Anonymity Level': metrics.get('anonymity_level', 0),
                'Precision (%)': 100 - metrics.get('precision_loss', 0),
                'Response Time (ms)': metrics.get('response_time', 0),
                'Privacy Score': self._calculate_privacy_score(metrics),
                'Utility Score': self._calculate_utility_score(metrics)
            })
        
        df = pd.DataFrame(comparison)
        print("\n" + df.to_string(index=False))
        
        return df
    
    @staticmethod
    def _calculate_privacy_score(metrics: Dict) -> float:
        """Calculate overall privacy score (0-100)"""
        # Higher is better
        reident_score = 100 - metrics.get('reidentification_risk', 50)
        anon_score = min(metrics.get('anonymity_level', 0) * 10, 100)
        return (reident_score + anon_score) / 2
    
    @staticmethod
    def _calculate_utility_score(metrics: Dict) -> float:
        """Calculate overall utility score (0-100)"""
        # Higher is better
        precision = 100 - metrics.get('precision_loss', 0)
        speed_score = max(0, 100 - metrics.get('response_time', 100) / 2)
        return (precision + speed_score) / 2
    
    def plot_radar_chart(self, comparison_df: pd.DataFrame):
        """Create radar chart comparing techniques"""
        print("\n📈 Generating Radar Chart...")
        
        categories = ['Privacy Score', 'Utility Score', 'Anonymity Level', 
                     'Precision (%)', 'Speed']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            if idx >= 3:  # Limit to 3 techniques
                break
            
            values = [
                row['Privacy Score'],
                row['Utility Score'],
                row['Anonymity Level'],
                row['Precision (%)'],
                100 - row['Response Time (ms)'] / 2  # Normalize speed
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Technique'], 
                   color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_title('Privacy-LBS Technique Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('technique_radar_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: technique_radar_comparison.png")
        plt.show()
    
    def plot_tradeoff_analysis(self, comparison_df: pd.DataFrame):
        """Create privacy-utility tradeoff plot"""
        print("\n📈 Generating Privacy-Utility Tradeoff Plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            ax.scatter(row['Utility Score'], row['Privacy Score'], 
                      s=300, alpha=0.6, color=colors[idx % len(colors)],
                      edgecolors='black', linewidth=2)
            ax.annotate(row['Technique'], 
                       (row['Utility Score'], row['Privacy Score']),
                       fontsize=10, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Utility Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Privacy Score', fontsize=12, fontweight='bold')
        ax.set_title('Privacy-Utility Tradeoff Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Add diagonal line (ideal balance)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Balance')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: privacy_utility_tradeoff.png")
        plt.show()

# ============================================================================
# ADVANCED VISUALIZATIONS
# ============================================================================

class AdvancedVisualizer:
    """Creates advanced visualization dashboards"""
    
    @staticmethod
    def create_heatmap(location_df: pd.DataFrame, title: str = "Location Density Heatmap"):
        """Create location density heatmap"""
        print(f"\n📈 Generating {title}...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            location_df['longitude'], location_df['latitude'],
            bins=50
        )
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                      cmap='YlOrRd', aspect='auto', alpha=0.8)
        
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Query Density')
        plt.tight_layout()
        
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.show()
    
    @staticmethod
    def create_performance_dashboard(metrics_dict: Dict):
        """Create comprehensive performance dashboard"""
        print("\n📈 Generating Performance Dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        techniques = list(metrics_dict.keys())
        
        # 1. Re-identification Risk
        ax1 = fig.add_subplot(gs[0, 0])
        risks = [m['reidentification_risk'] for m in metrics_dict.values()]
        bars1 = ax1.bar(techniques, risks, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_title('Re-identification Risk', fontweight='bold')
        ax1.set_ylabel('Risk (%)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, risks):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
        # 2. Information Loss
        ax2 = fig.add_subplot(gs[0, 1])
        losses = [m['information_loss'] for m in metrics_dict.values()]
        bars2 = ax2.bar(techniques, losses, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_title('Information Loss', fontweight='bold')
        ax2.set_ylabel('Distance (m)')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 10,
                    f'{val:.0f}m', ha='center', fontsize=9, fontweight='bold')
        
        # 3. Anonymity Level
        ax3 = fig.add_subplot(gs[0, 2])
        levels = [m['anonymity_level'] for m in metrics_dict.values()]
        bars3 = ax3.bar(techniques, levels, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax3.set_title('Anonymity Level', fontweight='bold')
        ax3.set_ylabel('Level')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars3, levels):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
        
        # 4. Response Time
        ax4 = fig.add_subplot(gs[1, 0])
        times = [m.get('response_time', 0) for m in metrics_dict.values()]
        bars4 = ax4.bar(techniques, times, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax4.set_title('Response Time', fontweight='bold')
        ax4.set_ylabel('Time (ms)')
        ax4.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars4, times):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 5,
                    f'{val:.0f}ms', ha='center', fontsize=9, fontweight='bold')
        
        # 5. Precision
        ax5 = fig.add_subplot(gs[1, 1])
        precisions = [100 - m.get('precision_loss', 0) for m in metrics_dict.values()]
        bars5 = ax5.bar(techniques, precisions, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax5.set_title('Service Precision', fontweight='bold')
        ax5.set_ylabel('Precision (%)')
        ax5.set_ylim(85, 100)
        ax5.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars5, precisions):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
        # 6. Overall Scores
        ax6 = fig.add_subplot(gs[1, 2])
        privacy_scores = [100 - r for r in risks]
        utility_scores = precisions
        x = np.arange(len(techniques))
        width = 0.35
        ax6.bar(x - width/2, privacy_scores, width, label='Privacy', color='#2ecc71')
        ax6.bar(x + width/2, utility_scores, width, label='Utility', color='#3498db')
        ax6.set_title('Privacy vs Utility', fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_xticks(x)
        ax6.set_xticklabels(techniques, rotation=45)
        ax6.legend()
        
        # 7-9. Summary Text
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary_text = "PERFORMANCE SUMMARY\n" + "="*80 + "\n\n"
        summary_text += f"Best Privacy Protection: {techniques[np.argmin(risks)]}\n"
        summary_text += f"Best Service Quality: {techniques[np.argmax(precisions)]}\n"
        summary_text += f"Fastest Response: {techniques[np.argmin(times)]}\n"
        summary_text += f"Highest Anonymity: {techniques[np.argmax(levels)]}\n"
        
        ax7.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Privacy-LBS Performance Dashboard', fontsize=16, fontweight='bold')
        
        plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: performance_dashboard.png")
        plt.show()

# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

def run_complete_analysis():
    """Run complete analysis pipeline"""
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS & VISUALIZATION")
    print("="*80)
    
    # Load data (assuming main script has been run)
    try:
        df = pd.read_csv('location_dataset.csv')
        print(f"\n✓ Loaded dataset: {len(df)} records")
    except FileNotFoundError:
        print("\n❌ Error: Run main implementation first to generate data!")
        return
    
    # Simulate attacks
    print("\n" + "="*80)
    print("PRIVACY ATTACK SIMULATION")
    print("="*80)
    
    attacker = PrivacyAttackSimulator()
    
    # Create mock anonymized data for attack simulation
    anon_df = df.copy()
    anon_df['latitude'] += np.random.normal(0, 0.01, len(anon_df))
    anon_df['longitude'] += np.random.normal(0, 0.01, len(anon_df))
    
    attacker.simulate_linkage_attack(df, anon_df)
    attacker.simulate_trajectory_attack(df)
    attacker.simulate_inference_attack(df)
    
    attack_report = attacker.generate_attack_report()
    print("\n" + attack_report)
    
    with open('attack_simulation_report.txt', 'w') as f:
        f.write(attack_report)
    print("✓ Saved: attack_simulation_report.txt")
    
    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Mock metrics from case study
    results_dict = {
        'k-anonymity': {
            'reidentification_risk': 3.2,
            'information_loss': 450,
            'anonymity_level': 10,
            'precision_loss': 6.5,
            'response_time': 198
        },
        'spatial-cloaking': {
            'reidentification_risk': 5.8,
            'information_loss': 520,
            'anonymity_level': 8,
            'precision_loss': 7.2,
            'response_time': 215
        },
        'geo-indistinguishability': {
            'reidentification_risk': 2.1,
            'information_loss': 380,
            'anonymity_level': 12,
            'precision_loss': 8.5,
            'response_time': 230
        }
    }
    
    analyzer = ComparativeAnalyzer()
    comparison_df = analyzer.compare_techniques(results_dict)
    
    # Save comparison
    comparison_df.to_csv('technique_comparison.csv', index=False)
    print("\n✓ Saved: technique_comparison.csv")
    
    # Generate visualizations
    analyzer.plot_radar_chart(comparison_df)
    analyzer.plot_tradeoff_analysis(comparison_df)
    
    # Advanced visualizations
    print("\n" + "="*80)
    print("ADVANCED VISUALIZATIONS")
    print("="*80)
    
    visualizer = AdvancedVisualizer()
    visualizer.create_heatmap(df, "Original Location Density Heatmap")
    visualizer.create_heatmap(anon_df, "Anonymized Location Density Heatmap")
    visualizer.create_performance_dashboard(results_dict)
    
    print("\n\n✅ COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. attack_simulation_report.txt")
    print("  2. technique_comparison.csv")
    print("  3. technique_radar_comparison.png")
    print("  4. privacy_utility_tradeoff.png")
    print("  5. original_location_density_heatmap.png")
    print("  6. anonymized_location_density_heatmap.png")
    print("  7. performance_dashboard.png")
    print("="*80)

if __name__ == "__main__":
    run_complete_analysis()