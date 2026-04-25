"""Publication-quality explainability visualizations for deception detection."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class ExplainabilityVisualizer:
    """Publication-quality explainability visualizations."""

    def __init__(self):
        """Set up matplotlib style for publication plots.
        
        Configures seaborn style, font sizes, and color palette.
        """
        # Use seaborn style with white background
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        })
        
        # Define colorblind-friendly palette
        self.cb_palette = sns.color_palette("colorblind")
        self.importance_color = self.cb_palette[0]  # blue
        self.shap_color = self.cb_palette[2]        # red
        self.truth_color = self.cb_palette[1]       # green
        self.deception_color = self.cb_palette[3]   # orange
        
        # Store current figure reference
        self._current_fig = None

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart comparing model importance vs SHAP importance.
        
        Args:
            importance_df: DataFrame with columns ['feature', 'importance', 
                'shap_importance', 'combined_score', 'rank', 'interpretation'].
            top_k: Number of top features to display (default 15).
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If required columns are missing or DataFrame is empty.
        """
        if importance_df.empty:
            raise ValueError("importance_df is empty")
        
        required_cols = {'feature', 'importance', 'shap_importance'}
        missing = required_cols - set(importance_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Sort by combined_score if available, else by importance
        if 'combined_score' in importance_df.columns:
            df = importance_df.sort_values('combined_score', ascending=False).head(top_k)
        else:
            df = importance_df.sort_values('importance', ascending=False).head(top_k)
        
        # Ensure we have at least one feature
        if df.empty:
            warnings.warn("No features to plot after filtering")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No feature importance data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            if save_path:
                fig.savefig(save_path, dpi=300)
            return fig
        
        # Create horizontal positions
        y_pos = np.arange(len(df))
        bar_height = 0.35
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
        
        # Plot side-by-side bars
        bars1 = ax.barh(y_pos - bar_height/2, df['importance'], 
                       height=bar_height, color=self.importance_color, 
                       label='Model Importance', edgecolor='black', linewidth=0.5)
        bars2 = ax.barh(y_pos + bar_height/2, df['shap_importance'], 
                       height=bar_height, color=self.shap_color, 
                       label='SHAP Importance', edgecolor='black', linewidth=0.5)
        
        # Add feature names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'])
        
        # Add value labels on bars
        for bar in bars1:
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', va='center', ha='left', fontsize=8)
        
        for bar in bars2:
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', va='center', ha='left', fontsize=8)
        
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Comparison (Model vs SHAP)')
        ax.legend(loc='lower right')
        
        # Add interpretation as hover-like annotations (optional)
        if 'interpretation' in df.columns:
            # Add interpretation as text on the right side if space
            max_x = max(df['importance'].max(), df['shap_importance'].max()) * 1.3
            for i, (_, row) in enumerate(df.iterrows()):
                if isinstance(row['interpretation'], str) and row['interpretation']:
                    ax.text(max_x, i, f"  {row['interpretation']}", 
                           va='center', fontsize=7, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5))
        
        ax.invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        X: np.ndarray,
        top_k: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """SHAP beeswarm-style summary plot.
        
        Each dot represents one sample, colored by feature value, 
        x-position is SHAP value.
        
        Args:
            shap_values: 2D array of SHAP values (n_samples, n_features).
            feature_names: List of feature names.
            X: 2D array of feature values (n_samples, n_features).
            top_k: Number of top features to display (default 15).
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If dimensions mismatch.
        """
        if shap_values.shape[1] != len(feature_names):
            raise ValueError("shap_values columns must match feature_names length")
        if X.shape[1] != len(feature_names):
            raise ValueError("X columns must match feature_names length")
        
        n_samples, n_features = shap_values.shape
        if n_samples == 0 or n_features == 0:
            warnings.warn("No data to plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No SHAP data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            if save_path:
                fig.savefig(save_path, dpi=300)
            return fig
        
        # Compute mean absolute SHAP to rank features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        # Prepare data for plotting
        top_shap = shap_values[:, top_indices]
        top_X = X[:, top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.5)))
        
        # Create colormap for feature values
        cmap = plt.cm.coolwarm
        norm = Normalize(vmin=np.nanmin(top_X), vmax=np.nanmax(top_X))
        
        # Plot each feature
        y_positions = np.arange(len(top_names))
        
        for i, (name, shap_vals, x_vals) in enumerate(zip(top_names, top_shap.T, top_X.T)):
            y = y_positions[i]
            
            # Add jitter for visibility
            jitter = np.random.normal(0, 0.05, size=len(shap_vals))
            
            # Scatter plot
            scatter = ax.scatter(
                shap_vals,
                y + jitter,
                c=x_vals,
                cmap=cmap,
                norm=norm,
                s=20,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.3,
            )
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label('Feature Value', fontsize=9)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Set labels and ticks
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_names)
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.set_ylabel('Feature')
        ax.set_title(f'SHAP Summary Plot (Top {top_k} Features)')
        
        # Add mean absolute SHAP annotation
        for i, idx in enumerate(top_indices):
            ax.text(0.02, y_positions[i], f'mean|SHAP|={mean_abs_shap[idx]:.3f}',
                   transform=ax.get_yaxis_transform(),
                   fontsize=7, va='center', bbox=dict(boxstyle='round,pad=0.2', 
                                                      facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_shap_waterfall(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        base_value: float,
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Waterfall plot for a single prediction's SHAP explanation.
        
        Red bars = features pushing toward deception (positive SHAP)
        Blue bars = features pushing toward truth (negative SHAP)
        
        Args:
            shap_values: 1D array of SHAP values for a single sample.
            feature_names: List of feature names.
            base_value: Base (expected) value from SHAP explainer.
            top_k: Number of top features to display (default 10).
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if shap_values.ndim != 1:
            if shap_values.ndim == 2 and shap_values.shape[0] == 1:
                shap_values = shap_values[0]
            else:
                raise ValueError("shap_values must be 1D or 2D with single sample")
        
        if len(shap_values) != len(feature_names):
            raise ValueError("shap_values length must match feature_names")
        
        # Select top K features by absolute SHAP
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        # Sort by SHAP value (deception to truth)
        sorted_indices = top_indices[np.argsort(shap_values[top_indices])[::-1]]
        
        top_shap = shap_values[sorted_indices]
        top_names = [feature_names[i] for i in sorted_indices]
        
        # Compute cumulative values for waterfall
        cumulative = base_value + np.cumsum(top_shap)
        cumulative = np.insert(cumulative, 0, base_value)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.6)))
        
        # Plot bars
        for i, (shap_val, name) in enumerate(zip(top_shap, top_names)):
            color = self.deception_color if shap_val > 0 else self.truth_color
            ax.barh(i, shap_val, left=cumulative[i], color=color, 
                   edgecolor='black', linewidth=0.5, height=0.8)
        
        # Add connecting lines
        for i in range(len(cumulative) - 1):
            ax.plot([cumulative[i], cumulative[i+1]], [i, i], 
                   color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        # Add base value and final prediction
        final_value = cumulative[-1]
        ax.axvline(x=base_value, color='black', linestyle=':', linewidth=1, 
                  label=f'Base Value: {base_value:.3f}')
        ax.axvline(x=final_value, color='red', linestyle=':', linewidth=1, 
                  label=f'Final Prediction: {final_value:.3f}')
        
        # Add feature labels
        ax.set_yticks(np.arange(len(top_names)))
        ax.set_yticklabels(top_names)
        
        # Add value annotations
        for i, (shap_val, cum_val) in enumerate(zip(top_shap, cumulative[1:])):
            ax.text(cum_val, i, f'{shap_val:+.3f}', 
                   va='center', ha='left' if shap_val > 0 else 'right',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Model Output (Deception Probability)')
        ax.set_title('SHAP Waterfall Explanation')
        ax.legend(loc='upper right')
        
        # Add interpretation
        ax.text(0.02, 0.98, f'Base: {base_value:.3f} → Final: {final_value:.3f}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_group_contributions(
        self,
        group_contributions: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Donut chart of feature group contributions.
        
        Args:
            group_contributions: Maps group name to normalized contribution (sums to 1.0).
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if not group_contributions:
            warnings.warn("Empty group_contributions dictionary")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, "No group contribution data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            if save_path:
                fig.savefig(save_path, dpi=300)
            return fig
        
        # Normalize contributions to sum to 1.0
        total = sum(group_contributions.values())
        if total == 0:
            # Equal distribution if all zero
            groups = list(group_contributions.keys())
            contributions = [1.0 / len(groups)] * len(groups)
        else:
            groups = list(group_contributions.keys())
            contributions = [group_contributions[g] / total for g in groups]
        
        # Sort by contribution
        sorted_indices = np.argsort(contributions)[::-1]
        groups = [groups[i] for i in sorted_indices]
        contributions = [contributions[i] for i in sorted_indices]
        
        # Create donut chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use colorblind-friendly palette
        colors = sns.color_palette("colorblind", len(groups))
        
        # Outer pie
        wedges, texts, autotexts = ax.pie(
            contributions,
            labels=groups,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 9},
        )
        
        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Equal aspect ratio
        ax.axis('equal')
        
        # Add title
        ax.set_title('Feature Group Contributions', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        plot_type: str = 'bar',
        top_k: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """SHAP summary plot (bar or beeswarm).
        
        Args:
            shap_values: 2D array of SHAP values (n_samples, n_features).
            feature_names: List of feature names.
            plot_type: 'bar' for bar chart of mean absolute SHAP,
                       'beeswarm' for beeswarm plot (requires X).
            top_k: Number of top features to display (default 15).
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If dimensions mismatch or unsupported plot_type.
        """
        if shap_values.shape[1] != len(feature_names):
            raise ValueError("shap_values columns must match feature_names length")
        
        n_samples, n_features = shap_values.shape
        if n_samples == 0 or n_features == 0:
            warnings.warn("No data to plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No SHAP data",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            if save_path:
                fig.savefig(save_path, dpi=300)
            return fig
        
        # Compute mean absolute SHAP to rank features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_means = mean_abs_shap[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        if plot_type == 'bar':
            fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.5)))
            y_pos = np.arange(len(top_names))
            ax.barh(y_pos, top_means, color=self.shap_color, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title(f'SHAP Feature Importance (Top {top_k})')
            ax.invert_yaxis()
            plt.tight_layout()
        elif plot_type == 'beeswarm':
            # Requires X (feature values) which we don't have; fallback to bar
            warnings.warn("Beeswarm plot requires X matrix; falling back to bar plot.")
            return self.plot_shap_summary(shap_values, feature_names, plot_type='bar', top_k=top_k, save_path=save_path)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_feature_groups(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_groups: Dict[str, List[str]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Grouped feature contribution plot.
        
        Args:
            shap_values: 2D array of SHAP values (n_samples, n_features).
            feature_names: List of feature names.
            feature_groups: Mapping from group name to list of feature names.
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        # Aggregate per group
        group_sums = {}
        for group, features in feature_groups.items():
            total = 0.0
            for feat in features:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    total += mean_abs_shap[idx]
            group_sums[group] = total
        
        # Normalize to percentages
        total = sum(group_sums.values())
        if total == 0:
            group_percents = {k: 0.0 for k in group_sums.keys()}
        else:
            group_percents = {k: v / total * 100 for k, v in group_sums.items()}
        
        # Sort groups by contribution
        sorted_groups = sorted(group_percents.items(), key=lambda x: x[1], reverse=True)
        groups = [g for g, _ in sorted_groups]
        percents = [p for _, p in sorted_groups]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(groups) * 0.5)))
        y_pos = np.arange(len(groups))
        colors = sns.color_palette("colorblind", len(groups))
        ax.barh(y_pos, percents, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel('Contribution (%)')
        ax.set_title('Feature Group Contributions (SHAP)')
        # Add percentage labels
        for i, p in enumerate(percents):
            ax.text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=9)
        
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_behavioral_interpretation(
        self,
        behavioral_explanations: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Human-readable behavioral interpretation visualization.
        
        Args:
            behavioral_explanations: List of dicts with keys 'feature', 'value',
                                     'shap_value', 'direction', 'interpretation'.
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if not behavioral_explanations:
            warnings.warn("Empty behavioral_explanations list")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No behavioral explanation data",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            if save_path:
                fig.savefig(save_path, dpi=300)
            return fig
        
        # Limit to top 10 for readability
        top_explanations = behavioral_explanations[:10]
        n = len(top_explanations)
        
        fig, ax = plt.subplots(figsize=(12, max(6, n * 0.8)))
        ax.axis('off')
        
        # Create a table-like visualization
        y_pos = np.arange(n) * 1.2
        for i, exp in enumerate(top_explanations):
            feature = exp.get('feature', 'Unknown')
            value = exp.get('value', 0.0)
            shap_val = exp.get('shap_value', 0.0)
            direction = exp.get('direction', 'neutral')
            interpretation = exp.get('interpretation', '')
            
            # Determine color based on direction
            if direction == 'toward_deception':
                color = self.deception_color
                dir_text = '→ Deception'
            elif direction == 'toward_truth':
                color = self.truth_color
                dir_text = '→ Truth'
            else:
                color = 'gray'
                dir_text = 'Neutral'
            
            # Feature name and value
            ax.text(0.1, y_pos[i], f'{feature}', fontsize=10, fontweight='bold',
                    verticalalignment='center')
            ax.text(0.3, y_pos[i], f'{value:.3f}', fontsize=9,
                    verticalalignment='center')
            # SHAP value with sign
            shap_sign = '+' if shap_val >= 0 else ''
            ax.text(0.45, y_pos[i], f'{shap_sign}{shap_val:.3f}', fontsize=9,
                    color=color, verticalalignment='center')
            # Direction arrow
            ax.text(0.6, y_pos[i], dir_text, fontsize=9, color=color,
                    verticalalignment='center')
            # Interpretation (wrap)
            ax.text(0.75, y_pos[i], interpretation, fontsize=8, wrap=True,
                    verticalalignment='center', style='italic')
        
        # Add column headers
        ax.text(0.1, y_pos[-1] + 1.5, 'Feature', fontsize=10, fontweight='bold')
        ax.text(0.3, y_pos[-1] + 1.5, 'Value', fontsize=10, fontweight='bold')
        ax.text(0.45, y_pos[-1] + 1.5, 'SHAP', fontsize=10, fontweight='bold')
        ax.text(0.6, y_pos[-1] + 1.5, 'Direction', fontsize=10, fontweight='bold')
        ax.text(0.75, y_pos[-1] + 1.5, 'Interpretation', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, y_pos[-1] + 3)
        ax.set_title('Behavioral Interpretation', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig

    def plot_explanation_dashboard(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_groups: Dict[str, List[str]],
        behavioral_explanations: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Combined dashboard with all plots.
        
        Args:
            shap_values: 2D array of SHAP values (n_samples, n_features).
            feature_names: List of feature names.
            feature_groups: Mapping from group name to list of feature names.
            behavioral_explanations: List of behavioral explanation dicts.
            save_path: Optional path to save figure at 300 DPI.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. SHAP summary bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_k = 10
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_means = mean_abs_shap[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        y_pos = np.arange(len(top_names))
        ax1.barh(y_pos, top_means, color=self.shap_color, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_names)
        ax1.set_xlabel('Mean |SHAP|')
        ax1.set_title('SHAP Feature Importance (Top 10)')
        ax1.invert_yaxis()
        
        # 2. Feature group contributions
        ax2 = fig.add_subplot(gs[0, 1])
        group_sums = {}
        for group, features in feature_groups.items():
            total = 0.0
            for feat in features:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    total += mean_abs_shap[idx]
            group_sums[group] = total
        total = sum(group_sums.values())
        if total == 0:
            group_percents = {k: 0.0 for k in group_sums.keys()}
        else:
            group_percents = {k: v / total * 100 for k, v in group_sums.items()}
        sorted_groups = sorted(group_percents.items(), key=lambda x: x[1], reverse=True)
        groups = [g for g, _ in sorted_groups]
        percents = [p for _, p in sorted_groups]
        colors = sns.color_palette("colorblind", len(groups))
        ax2.barh(np.arange(len(groups)), percents, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(np.arange(len(groups)))
        ax2.set_yticklabels(groups)
        ax2.set_xlabel('Contribution (%)')
        ax2.set_title('Feature Group Contributions')
        ax2.invert_yaxis()
        
        # 3. Behavioral interpretation table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        if behavioral_explanations:
            top_explanations = behavioral_explanations[:5]
            y_pos = np.arange(len(top_explanations)) * 1.5
            for i, exp in enumerate(top_explanations):
                feature = exp.get('feature', 'Unknown')
                shap_val = exp.get('shap_value', 0.0)
                direction = exp.get('direction', 'neutral')
                interpretation = exp.get('interpretation', '')
                color = self.deception_color if direction == 'toward_deception' else self.truth_color
                ax3.text(0.1, y_pos[i], f'{feature}', fontsize=10, fontweight='bold')
                ax3.text(0.3, y_pos[i], f'{shap_val:+.3f}', fontsize=9, color=color)
                ax3.text(0.5, y_pos[i], direction, fontsize=9, color=color)
                ax3.text(0.7, y_pos[i], interpretation, fontsize=8, wrap=True, style='italic')
            ax3.set_title('Top Behavioral Indicators', fontsize=12, pad=10)
        else:
            ax3.text(0.5, 0.5, 'No behavioral explanations', ha='center', va='center', fontsize=12)
        
        fig.suptitle('Explainability Dashboard', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        self._current_fig = fig
        return fig