"""Visualization tools for deception detection evaluation.

This module provides plotting functions for confusion matrices, ROC curves,
precision‑recall curves, cross‑validation results, ablation studies, and
subject‑wise performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Create publication‑quality plots for evaluation results.

    All methods accept a `save_path` argument; if provided, the figure is saved
    to that path. Otherwise, the figure is returned for further customization.

    Attributes:
        style: Matplotlib style to use (default 'seaborn‑whitegrid').
        figsize: Default figure size as (width, height).
        dpi: Resolution for saved figures.
    """

    def __init__(
        self,
        style: str = "seaborn‑whitegrid",
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 150,
    ) -> None:
        """Initialize visualizer.

        Args:
            style: Matplotlib style name.
            figsize: Default figure size (width, height) in inches.
            dpi: Dots per inch for saved figures.
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use(style)

    def plot_confusion_matrix(
        self,
        cm: Union[np.ndarray, List[List[int]]],
        labels: List[str],
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot a confusion matrix.

        Args:
            cm: 2×2 or N×N confusion matrix.
            labels: List of class labels (e.g., ['Truthful', 'Deceptive']).
            normalize: If True, show proportions instead of counts.
            title: Plot title.
            cmap: Colormap for the heatmap.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        cm = np.asarray(cm)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Normalized frequency" if normalize else "Count"},
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
        label: str = "Model",
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot ROC curve with AUC.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted probabilities for the positive class.
            label: Label for the curve (used in legend).
            title: Plot title.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved ROC curve to {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
        label: str = "Model",
        title: str = "Precision‑Recall Curve",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot precision‑recall curve with average precision.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted probabilities for the positive class.
            label: Label for the curve (used in legend).
            title: Plot title.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(recall, precision, lw=2, label=f"{label} (AP = {avg_precision:.3f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved precision‑recall curve to {save_path}")

        return fig

    def plot_cross_validation_results(
        self,
        cv_results: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        title: str = "Cross‑Validation Performance",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot box‑whisker or bar chart of metrics across CV folds.

        Args:
            cv_results: Dictionary as returned by CrossValidator.run_cross_validation.
                Must contain 'per_fold_metrics' (list of dicts).
            metrics: List of metric names to plot. If None, defaults to
                ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'].
            title: Plot title.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        if "per_fold_metrics" not in cv_results:
            raise ValueError("cv_results must contain 'per_fold_metrics'")

        per_fold = cv_results["per_fold_metrics"]
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
            # Add AUC if present in any fold
            if any("auc_roc" in fold for fold in per_fold):
                metrics.append("auc_roc")

        # Build DataFrame of metric values across folds
        data = []
        for fold_idx, fold_metrics in enumerate(per_fold):
            for metric in metrics:
                if metric in fold_metrics:
                    data.append(
                        {
                            "fold": f"Fold {fold_idx+1}",
                            "metric": metric,
                            "value": fold_metrics[metric],
                        }
                    )

        if not data:
            logger.warning("No metric data to plot")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.boxplot(data=df, x="metric", y="value", ax=ax, palette="Set2")
        sns.stripplot(
            data=df,
            x="metric",
            y="value",
            ax=ax,
            color="black",
            alpha=0.7,
            jitter=True,
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved cross‑validation results plot to {save_path}")

        return fig

    def plot_ablation_results(
        self,
        ablation_results: Dict[str, Dict[str, Any]],
        metric: str = "accuracy",
        title: str = "Ablation Study",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot bar chart comparing performance across ablation conditions.

        Args:
            ablation_results: Dictionary as returned by AblationStudy.run_ablation.
                Keys are condition names, each containing 'mean_metrics' dict.
            metric: Metric to compare across conditions.
            title: Plot title.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        conditions = []
        means = []
        stds = []

        for cond_name, cond_data in ablation_results.items():
            if "mean_metrics" in cond_data and metric in cond_data["mean_metrics"]:
                conditions.append(cond_name.replace("_", " ").title())
                means.append(cond_data["mean_metrics"][metric])
                stds.append(cond_data.get("std_metrics", {}).get(metric, 0.0))

        if not conditions:
            logger.warning(f"Metric '{metric}' not found in any condition")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        x = np.arange(len(conditions))
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on top of bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved ablation results plot to {save_path}")

        return fig

    def plot_subject_performance(
        self,
        per_subject_metrics: Dict[str, Dict[str, Any]],
        metric: str = "accuracy",
        title: str = "Subject‑wise Performance",
        sort_by: str = "value",  # 'value' or 'subject'
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot bar chart of a chosen metric across subjects.

        Args:
            per_subject_metrics: Dictionary mapping subject_id -> metrics dict.
            metric: Metric to plot (must exist in each subject's dict).
            title: Plot title.
            sort_by: 'value' to sort bars by metric value, 'subject' to keep original order.
            save_path: If provided, save figure to this path.

        Returns:
            Matplotlib Figure object.
        """
        subjects = []
        values = []

        for subj, metrics in per_subject_metrics.items():
            if metric in metrics:
                subjects.append(subj)
                values.append(metrics[metric])
            else:
                logger.warning(f"Metric '{metric}' not found for subject {subj}")

        if not subjects:
            logger.warning("No subject data to plot")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        if sort_by == "value":
            # Sort by value descending
            sorted_pairs = sorted(zip(subjects, values), key=lambda x: x[1], reverse=True)
            subjects, values = zip(*sorted_pairs)
        # else keep original order

        x = np.arange(len(subjects))
        fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.5), 6))
        bars = ax.bar(x, values, color="teal", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=90, ha="center")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved subject performance plot to {save_path}")

        return fig
