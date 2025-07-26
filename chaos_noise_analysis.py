#!/usr/bin/env python3
"""
Distinguishing Noise from Chaos Analysis for USD/EUR Exchange Rates

Based on the methodology from:
"Distinguishing Noise from Chaos" by Rosso et al. (2007)
Physical Review Letters 99, 154102

This script implements the complexity-entropy causality plane analysis
to determine whether USD/EUR exchange rate time series exhibit
chaotic or stochastic behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import permutations
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BandtPompeAnalysis:
    """
    Implementation of Bandt-Pompe ordinal pattern analysis
    for distinguishing chaos from noise.
    """
    
    def __init__(self, embedding_dimension: int = 6):
        """
        Initialize the Bandt-Pompe analyzer.
        
        Args:
            embedding_dimension: Embedding dimension D (typically 3-7)
        """
        self.D = embedding_dimension
        self.factorial_D = np.math.factorial(self.D)
        
    def ordinal_patterns(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract ordinal patterns from time series using Bandt-Pompe method.
        
        Args:
            time_series: 1D array of time series data
            
        Returns:
            Array of ordinal pattern indices
        """
        N = len(time_series)
        patterns = []
        
        # Generate all possible permutations for embedding dimension D
        all_perms = list(permutations(range(self.D)))
        perm_to_index = {perm: i for i, perm in enumerate(all_perms)}
        
        # Extract ordinal patterns
        for i in range(N - self.D + 1):
            # Get D consecutive values
            segment = time_series[i:i + self.D]
            
            # Get sorting indices (ordinal pattern)
            sorted_indices = np.argsort(segment)
            
            # Convert to tuple for dictionary lookup
            pattern = tuple(sorted_indices)
            
            # Map pattern to index
            pattern_index = perm_to_index[pattern]
            patterns.append(pattern_index)
            
        return np.array(patterns)
    
    def probability_distribution(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution of ordinal patterns.
        
        Args:
            time_series: 1D array of time series data
            
        Returns:
            Probability distribution P = {p_π}
        """
        patterns = self.ordinal_patterns(time_series)
        
        # Count occurrences of each pattern
        counts = np.bincount(patterns, minlength=self.factorial_D)
        
        # Convert to probabilities
        probabilities = counts / len(patterns)
        
        return probabilities
    
    def shannon_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Normalized Shannon entropy H_S
        """
        # Remove zero probabilities for log calculation
        p_nonzero = probabilities[probabilities > 0]
        
        if len(p_nonzero) == 0:
            return 0.0
            
        # Calculate Shannon entropy
        S = -np.sum(p_nonzero * np.log(p_nonzero))
        
        # Normalize by maximum entropy
        S_max = np.log(self.factorial_D)
        H_S = S / S_max if S_max > 0 else 0.0
        
        return H_S
    
    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        
        Args:
            p, q: Probability distributions
            
        Returns:
            Jensen-Shannon divergence
        """
        # Midpoint distribution
        m = 0.5 * (p + q)
        
        # KL divergences (handling zeros)
        def kl_div(x, y):
            mask = (x > 0) & (y > 0)
            if not np.any(mask):
                return 0.0
            return np.sum(x[mask] * np.log(x[mask] / y[mask]))
        
        js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return js_div
    
    def statistical_complexity(self, probabilities: np.ndarray) -> float:
        """
        Compute intensive statistical complexity measure C_JS.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Statistical complexity C_JS
        """
        # Uniform distribution
        uniform_dist = np.ones(self.factorial_D) / self.factorial_D
        
        # Jensen-Shannon divergence (disequilibrium)
        Q_J = self.jensen_shannon_divergence(probabilities, uniform_dist)
        
        # Normalize Q_J (Q_0 normalization constant)
        Q_0 = -2 * ((self.factorial_D + 1) / self.factorial_D) * np.log(self.factorial_D + 1) + 2 * np.log(2 * self.factorial_D)
        Q_J_normalized = Q_J / Q_0 if Q_0 > 0 else 0.0
        
        # Shannon entropy
        H_S = self.shannon_entropy(probabilities)
        
        # Statistical complexity
        C_JS = Q_J_normalized * H_S
        
        return C_JS
    
    def complexity_entropy_analysis(self, time_series: np.ndarray) -> Tuple[float, float]:
        """
        Perform complete complexity-entropy analysis.
        
        Args:
            time_series: 1D array of time series data
            
        Returns:
            Tuple of (entropy H_S, complexity C_JS)
        """
        probabilities = self.probability_distribution(time_series)
        H_S = self.shannon_entropy(probabilities)
        C_JS = self.statistical_complexity(probabilities)
        
        return H_S, C_JS


class ChaosNoiseClassifier:
    """
    Classifier for distinguishing chaos from noise using the CH plane.
    """
    
    def __init__(self, embedding_dimension: int = 6):
        """
        Initialize the classifier.
        
        Args:
            embedding_dimension: Embedding dimension for Bandt-Pompe analysis
        """
        self.bp_analyzer = BandtPompeAnalysis(embedding_dimension)
        self.embedding_dim = embedding_dimension
        
    def analyze_time_series(self, data: np.ndarray, window_size: Optional[int] = None) -> dict:
        """
        Analyze time series for chaos vs noise characteristics.
        
        Args:
            data: Time series data
            window_size: Size of sliding window for temporal analysis
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'entropy': [],
            'complexity': [],
            'analysis_type': 'full_series'
        }
        
        if window_size is None:
            # Analyze full time series
            H_S, C_JS = self.bp_analyzer.complexity_entropy_analysis(data)
            results['entropy'] = [H_S]
            results['complexity'] = [C_JS]
            results['classification'] = self._classify_single(H_S, C_JS)
        else:
            # Sliding window analysis
            results['analysis_type'] = 'sliding_window'
            results['window_size'] = window_size
            
            for i in range(len(data) - window_size + 1):
                window_data = data[i:i + window_size]
                H_S, C_JS = self.bp_analyzer.complexity_entropy_analysis(window_data)
                results['entropy'].append(H_S)
                results['complexity'].append(C_JS)
            
            # Classify based on mean values
            mean_H_S = np.mean(results['entropy'])
            mean_C_JS = np.mean(results['complexity'])
            results['classification'] = self._classify_single(mean_H_S, mean_C_JS)
            results['mean_entropy'] = mean_H_S
            results['mean_complexity'] = mean_C_JS
            
        return results
    
    def _classify_single(self, entropy: float, complexity: float) -> str:
        """
        Classify a single (entropy, complexity) point.
        
        Args:
            entropy: Normalized Shannon entropy
            complexity: Statistical complexity
            
        Returns:
            Classification string
        """
        # Based on the paper's findings:
        # - Chaotic systems: 0.45 < H_S < 0.7, high C_JS (near maximum)
        # - Stochastic processes: Higher H_S, lower C_JS
        # - White noise: H_S ≈ 1, C_JS ≈ 0
        
        if entropy < 0.45:
            return "Low entropy (possibly periodic)"
        elif 0.45 <= entropy <= 0.7 and complexity > 0.4:
            return "Chaotic"
        elif entropy > 0.9 and complexity < 0.1:
            return "White noise"
        elif entropy > 0.7 and complexity < 0.3:
            return "Stochastic"
        else:
            return "Mixed/Transitional"


class ExchangeRateAnalyzer:
    """
    Main analyzer for USD/EUR exchange rate data.
    """
    
    def __init__(self, embedding_dimension: int = 6):
        """
        Initialize the exchange rate analyzer.
        
        Args:
            embedding_dimension: Embedding dimension for analysis
        """
        self.classifier = ChaosNoiseClassifier(embedding_dimension)
        self.data = None
        self.results = None
        
    def load_data(self, file_path: str, date_column: str = 'Date', 
                  rate_column: str = 'Rate') -> pd.DataFrame:
        """
        Load exchange rate data from file.
        
        Args:
            file_path: Path to data file (CSV, Excel, etc.)
            date_column: Name of date column
            rate_column: Name of exchange rate column
            
        Returns:
            Loaded DataFrame
        """
        # TODO: Implement data loading logic
        # This is a skeleton - you'll need to customize based on your data format
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Store the data
        self.data = df
        
        return df
    
    def preprocess_data(self, method: str = 'returns') -> np.ndarray:
        """
        Preprocess exchange rate data for analysis.
        
        Args:
            method: Preprocessing method ('returns', 'log_returns', 'raw', 'differenced')
            
        Returns:
            Preprocessed time series
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Assuming the rate column is the second column
        rates = self.data.iloc[:, 1].values
        
        if method == 'raw':
            processed = rates
        elif method == 'returns':
            processed = np.diff(rates) / rates[:-1]
        elif method == 'log_returns':
            processed = np.diff(np.log(rates))
        elif method == 'differenced':
            processed = np.diff(rates)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        # Remove any NaN or infinite values
        processed = processed[np.isfinite(processed)]
        
        return processed
    
    def run_analysis(self, preprocessing_method: str = 'log_returns',
                    window_size: Optional[int] = None) -> dict:
        """
        Run the complete chaos vs noise analysis.
        
        Args:
            preprocessing_method: How to preprocess the data
            window_size: Size for sliding window analysis (None for full series)
            
        Returns:
            Analysis results dictionary
        """
        # Preprocess data
        processed_data = self.preprocess_data(preprocessing_method)
        
        # Run analysis
        self.results = self.classifier.analyze_time_series(processed_data, window_size)
        self.results['preprocessing_method'] = preprocessing_method
        self.results['data_length'] = len(processed_data)
        
        return self.results
    
    def plot_ch_plane(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the complexity-entropy causality plane.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No analysis results. Run run_analysis() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot theoretical bounds (simplified)
        entropy_range = np.linspace(0, 1, 1000)
        
        # Maximum complexity curve (approximate)
        c_max = 0.5 * entropy_range * (1 - entropy_range)
        
        # Minimum complexity (always 0 for uniform and delta distributions)
        c_min = np.zeros_like(entropy_range)
        
        # Plot bounds
        ax.fill_between(entropy_range, c_min, c_max, alpha=0.2, color='gray', 
                       label='Feasible region')
        ax.plot(entropy_range, c_max, 'k--', alpha=0.7, label='C_max')
        ax.plot(entropy_range, c_min, 'k-', alpha=0.7, label='C_min')
        
        # Plot our data
        if self.results['analysis_type'] == 'full_series':
            ax.scatter(self.results['entropy'], self.results['complexity'], 
                      c='red', s=100, alpha=0.8, label='USD/EUR', zorder=5)
        else:
            # Sliding window analysis
            ax.scatter(self.results['entropy'], self.results['complexity'], 
                      c='red', alpha=0.6, s=20, label='USD/EUR windows')
            # Highlight mean
            ax.scatter(self.results['mean_entropy'], self.results['mean_complexity'],
                      c='darkred', s=150, alpha=1.0, label='Mean', zorder=5,
                      marker='*')
        
        # Add reference points for known systems (from the paper)
        reference_points = {
            'White noise': (0.95, 0.05),
            'Chaotic systems': (0.6, 0.4),
            'Periodic': (0.3, 0.0)
        }
        
        for name, (h, c) in reference_points.items():
            ax.scatter(h, c, marker='x', s=100, alpha=0.7, label=f'{name} (ref)')
        
        ax.set_xlabel('Normalized Shannon Entropy (H_S)')
        ax.set_ylabel('Statistical Complexity (C_JS)')
        ax.set_title('Complexity-Entropy Causality Plane\nUSD/EUR Exchange Rate Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        
        return fig
    
    def plot_time_evolution(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot time evolution of complexity and entropy (for sliding window analysis).
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.results is None or self.results['analysis_type'] != 'sliding_window':
            raise ValueError("No sliding window analysis results available.")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Time axis
        time_axis = range(len(self.results['entropy']))
        
        # Plot entropy evolution
        ax1.plot(time_axis, self.results['entropy'], 'b-', alpha=0.7)
        ax1.set_ylabel('Entropy (H_S)')
        ax1.set_title('Time Evolution of Chaos/Noise Indicators')
        ax1.grid(True, alpha=0.3)
        
        # Plot complexity evolution
        ax2.plot(time_axis, self.results['complexity'], 'r-', alpha=0.7)
        ax2.set_ylabel('Complexity (C_JS)')
        ax2.grid(True, alpha=0.3)
        
        # Plot ratio (complexity/entropy) as a discriminator
        ratio = np.array(self.results['complexity']) / np.array(self.results['entropy'])
        ax3.plot(time_axis, ratio, 'g-', alpha=0.7)
        ax3.set_ylabel('C_JS / H_S')
        ax3.set_xlabel('Window Index')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a text report of the analysis results.
        
        Returns:
            Formatted analysis report
        """
        if self.results is None:
            return "No analysis results available."
        
        report = []
        report.append("=" * 60)
        report.append("CHAOS vs NOISE ANALYSIS REPORT")
        report.append("USD/EUR Exchange Rate")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Data preprocessing: {self.results['preprocessing_method']}")
        report.append(f"Data length: {self.results['data_length']} points")
        report.append(f"Embedding dimension: {self.classifier.embedding_dim}")
        report.append("")
        
        if self.results['analysis_type'] == 'full_series':
            report.append("FULL SERIES ANALYSIS:")
            report.append(f"  Shannon Entropy (H_S): {self.results['entropy'][0]:.4f}")
            report.append(f"  Statistical Complexity (C_JS): {self.results['complexity'][0]:.4f}")
            report.append(f"  Classification: {self.results['classification']}")
        else:
            report.append("SLIDING WINDOW ANALYSIS:")
            report.append(f"  Window size: {self.results['window_size']}")
            report.append(f"  Number of windows: {len(self.results['entropy'])}")
            report.append(f"  Mean Shannon Entropy: {self.results['mean_entropy']:.4f}")
            report.append(f"  Mean Statistical Complexity: {self.results['mean_complexity']:.4f}")
            report.append(f"  Classification: {self.results['classification']}")
            report.append("")
            report.append("  Entropy statistics:")
            report.append(f"    Min: {np.min(self.results['entropy']):.4f}")
            report.append(f"    Max: {np.max(self.results['entropy']):.4f}")
            report.append(f"    Std: {np.std(self.results['entropy']):.4f}")
            report.append("")
            report.append("  Complexity statistics:")
            report.append(f"    Min: {np.min(self.results['complexity']):.4f}")
            report.append(f"    Max: {np.max(self.results['complexity']):.4f}")
            report.append(f"    Std: {np.std(self.results['complexity']):.4f}")
        
        report.append("")
        report.append("INTERPRETATION:")
        report.append("- Entropy ∈ [0.45, 0.7] + High complexity → Chaotic")
        report.append("- Entropy > 0.9 + Low complexity → Stochastic/Noise")
        report.append("- Entropy < 0.45 → Periodic/Regular")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Example usage of the USD/EUR chaos vs noise analyzer.
    """
    # Initialize analyzer
    analyzer = ExchangeRateAnalyzer(embedding_dimension=6)
    
    # TODO: Load your data
    # analyzer.load_data('path/to/your/usd_eur_data.csv')
    
    # Example with synthetic data for testing
    print("Creating synthetic USD/EUR data for demonstration...")
    
    # Generate synthetic exchange rate data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create a mixed signal: trend + noise + some chaotic component
    trend = 1.1 + 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = 0.02 * np.random.randn(len(dates))
    
    # Add a simple chaotic component (logistic map influence)
    chaotic = np.zeros(len(dates))
    x = 0.5
    for i in range(len(dates)):
        x = 3.9 * x * (1 - x)  # Chaotic logistic map
        chaotic[i] = 0.01 * (x - 0.5)
    
    rates = trend + noise + chaotic
    
    # Create DataFrame
    synthetic_data = pd.DataFrame({
        'Date': dates,
        'USD_EUR_Rate': rates
    })
    
    analyzer.data = synthetic_data
    
    print("Running analysis...")
    
    # Run full series analysis
    results_full = analyzer.run_analysis(preprocessing_method='log_returns')
    
    # Run sliding window analysis
    results_window = analyzer.run_analysis(
        preprocessing_method='log_returns',
        window_size=100
    )
    
    # Generate and print reports
    print("\nFULL SERIES ANALYSIS:")
    analyzer.results = results_full
    print(analyzer.generate_report())
    
    print("\nSLIDING WINDOW ANALYSIS:")
    analyzer.results = results_window
    print(analyzer.generate_report())
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # Plot CH plane for full series
    analyzer.results = results_full
    fig1 = analyzer.plot_ch_plane()
    fig1.suptitle('Full Series Analysis')
    
    # Plot CH plane for windowed analysis
    analyzer.results = results_window
    fig2 = analyzer.plot_ch_plane()
    fig2.suptitle('Sliding Window Analysis')
    
    # Plot time evolution
    fig3 = analyzer.plot_time_evolution()
    
    plt.show()


if __name__ == "__main__":
    main()
