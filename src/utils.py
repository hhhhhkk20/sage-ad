"""
SAGE-AD: Utility Functions Module
===================================
This module provides utility functions for:
1. Data loading and preprocessing
2. Cohort harmonization
3. Logging and visualization
4. Result saving and loading
"""

import json
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


class DataLoader:
    """
    Load and preprocess longitudinal survey data from different cohorts.
    """

    @staticmethod
    def load_elsa_data(file_path: str) -> pd.DataFrame:
        """
        Load ELSA (English Longitudinal Study of Ageing) data.

        Args:
            file_path: Path to ELSA data file

        Returns:
            Pandas DataFrame with harmonized ELSA data
        """
        # Pseudocode for loading ELSA data
        # In practice, this would involve reading CSV/STATA files and harmonizing variables
        # df = pd.read_stata(file_path) or pd.read_csv(file_path)
        # return DataLoader._harmonize_elsa(df)

        print(f"Loading ELSA data from {file_path}...")
        # Placeholder
        return pd.DataFrame()

    @staticmethod
    def load_hrs_data(file_path: str) -> pd.DataFrame:
        """
        Load HRS (Health and Retirement Study) data.

        Args:
            file_path: Path to HRS data file

        Returns:
            Pandas DataFrame with harmonized HRS data
        """
        print(f"Loading HRS data from {file_path}...")
        # Placeholder
        return pd.DataFrame()

    @staticmethod
    def load_share_data(file_path: str) -> pd.DataFrame:
        """
        Load SHARE (Survey of Health, Ageing and Retirement in Europe) data.

        Args:
            file_path: Path to SHARE data file

        Returns:
            Pandas DataFrame with harmonized SHARE data
        """
        print(f"Loading SHARE data from {file_path}...")
        # Placeholder
        return pd.DataFrame()

    @staticmethod
    def harmonize_variables(
        df: pd.DataFrame,
        cohort_name: str,
        variable_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Harmonize variables across cohorts.

        Args:
            df: Raw cohort data
            cohort_name: Name of cohort (ELSA, HRS, SHARE)
            variable_mapping: Dictionary mapping cohort-specific to common variable names

        Returns:
            Harmonized DataFrame
        """
        df_harmonized = df.copy()

        # Rename variables according to mapping
        df_harmonized = df_harmonized.rename(columns=variable_mapping)

        # Standardize cognitive assessments to z-scores
        cognitive_vars = [col for col in df_harmonized.columns if 'cognitive' in col.lower()]
        for var in cognitive_vars:
            if var in df_harmonized.columns:
                df_harmonized[f"{var}_z"] = (
                    df_harmonized[var] - df_harmonized[var].mean()
                ) / df_harmonized[var].std()

        return df_harmonized


class CohortProcessor:
    """
    Process cohort data to create temporal symptom profiles.
    """

    @staticmethod
    def identify_cases_and_controls(
        df: pd.DataFrame,
        ad_diagnosis_col: str = "ad_diagnosis",
        diagnosis_age_col: str = "diagnosis_age"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify AD cases and matched controls.

        Args:
            df: Cohort dataframe
            ad_diagnosis_col: Column indicating AD diagnosis
            diagnosis_age_col: Column indicating age at diagnosis

        Returns:
            Tuple of (cases_df, controls_df)
        """
        # Identify cases: participants who developed AD during follow-up
        cases = df[df[ad_diagnosis_col] == 1].copy()

        # Identify controls: participants who remained AD-free
        controls = df[df[ad_diagnosis_col] == 0].copy()

        # Sample controls at 1:3 ratio
        n_controls_needed = min(len(cases) * 3, len(controls))
        controls = controls.sample(n=n_controls_needed, random_state=42)

        return cases, controls

    @staticmethod
    def create_temporal_profiles(
        df: pd.DataFrame,
        participant_id_col: str = "participant_id",
        wave_col: str = "wave",
        age_col: str = "age"
    ) -> List:
        """
        Create temporal symptom profiles from longitudinal data.

        Args:
            df: Longitudinal cohort data
            participant_id_col: Column name for participant ID
            wave_col: Column name for survey wave
            age_col: Column name for age

        Returns:
            List of TemporalSymptomProfile objects
        """
        from llm_inference import TemporalSymptomProfile

        profiles = []

        # Group by participant
        for participant_id, group in df.groupby(participant_id_col):
            # Sort by wave/age
            group = group.sort_values(by=[wave_col, age_col])

            # Extract features by domain
            cognitive_features = []
            functional_features = []
            physiological_features = []
            neuropsychiatric_features = []

            for _, row in group.iterrows():
                # Extract cognitive features
                cognitive_features.append({
                    "word_recall_immediate": row.get("word_recall_immediate", None),
                    "word_recall_delayed": row.get("word_recall_delayed", None),
                    "orientation": row.get("orientation", None),
                    "serial_7s": row.get("serial_7s", None)
                })

                # Extract functional features
                functional_features.append({
                    "ADL_impairment": row.get("ADL_impairment", 0),
                    "IADL_impairment": row.get("IADL_impairment", 0)
                })

                # Extract physiological features
                physiological_features.append({
                    "BMI": row.get("BMI", None),
                    "grip_strength": row.get("grip_strength", None),
                    "gait_speed": row.get("gait_speed", None),
                    "systolic_bp": row.get("systolic_bp", None),
                    "diastolic_bp": row.get("diastolic_bp", None)
                })

                # Extract neuropsychiatric features
                neuropsychiatric_features.append({
                    "depression_score": row.get("depression_score", 0),
                    "anxiety": row.get("anxiety", "none")
                })

            # Create profile
            profile = TemporalSymptomProfile(
                participant_id=str(participant_id),
                age_at_assessments=group[age_col].tolist(),
                cognitive_features=cognitive_features,
                functional_features=functional_features,
                physiological_features=physiological_features,
                neuropsychiatric_features=neuropsychiatric_features,
                demographics={
                    "age": group[age_col].iloc[0],
                    "sex": group["sex"].iloc[0] if "sex" in group.columns else "Unknown",
                    "education": group["education_years"].iloc[0] if "education_years" in group.columns else None
                }
            )

            profiles.append(profile)

        return profiles


class ResultsManager:
    """
    Save and load experiment results.
    """

    @staticmethod
    def save_results(
        results: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> str:
        """
        Save experiment results to JSON file.

        Args:
            results: Dictionary containing results
            output_dir: Output directory
            experiment_name: Name of experiment

        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = ResultsManager._make_json_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath

    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """
        Load experiment results from JSON file.

        Args:
            filepath: Path to results file

        Returns:
            Dictionary containing results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)

        print(f"Results loaded from {filepath}")
        return results

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: ResultsManager._make_json_serializable(value)
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultsManager._make_json_serializable(item) for item in obj]
        else:
            return obj

    @staticmethod
    def save_model_cache(
        model_predictions: Dict[str, Any],
        cache_dir: str,
        model_name: str
    ) -> str:
        """
        Save model predictions to cache (pickle format for faster loading).

        Args:
            model_predictions: Predictions and metadata
            cache_dir: Cache directory
            model_name: Model identifier

        Returns:
            Path to cached file
        """
        os.makedirs(cache_dir, exist_ok=True)

        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        filename = f"cache_{safe_model_name}.pkl"
        filepath = os.path.join(cache_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(model_predictions, f)

        return filepath

    @staticmethod
    def load_model_cache(cache_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load cached model predictions.

        Args:
            cache_dir: Cache directory
            model_name: Model identifier

        Returns:
            Cached predictions or None if not found
        """
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        filename = f"cache_{safe_model_name}.pkl"
        filepath = os.path.join(cache_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'rb') as f:
            predictions = pickle.load(f)

        return predictions


class Logger:
    """
    Logging utility for tracking experiment progress.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
        """
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{experiment_name}_{timestamp}.log"
        self.log_path = os.path.join(log_dir, log_filename)

        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        """
        Write message to log file.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_path, 'a') as f:
            f.write(log_entry)

        # Also print to console
        print(log_entry.strip())

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        message = f"{prefix}Metrics:\n"
        for metric_name, value in metrics.items():
            message += f"  {metric_name}: {value:.4f}\n"

        self.log(message)


class ConfigManager:
    """
    Manage experiment configurations.
    """

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        return config

    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "models": [
                "gpt-4o",
                "gemini-2.5-flash",
                "claude-4.5-sonnet"
            ],
            "inference_strategies": ["zero_shot", "few_shot", "cot"],
            "prediction_horizons": [1, 2, 3, 4],
            "cohorts": ["ELSA", "HRS", "SHARE"],
            "bootstrap_iterations": 5000,
            "confidence_level": 0.95,
            "random_seed": 42,
            "output_dir": "./results",
            "cache_dir": "./cache",
            "log_dir": "./logs"
        }

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            config_path: Path to save config
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to {config_path}")


# Example usage
if __name__ == "__main__":
    # Example: Create default config
    config_manager = ConfigManager()
    default_config = config_manager.create_default_config()

    print("Default Configuration:")
    print(json.dumps(default_config, indent=2))

    # Example: Initialize logger
    logger = Logger(log_dir="./logs", experiment_name="test_experiment")
    logger.log("Starting experiment...")

    # Example: Save results
    mock_results = {
        "model": "gpt-4o",
        "metrics": {
            "accuracy": 0.75,
            "f1_score": 0.58,
            "precision": 0.48,
            "recall": 0.78
        },
        "timestamp": datetime.now().isoformat()
    }

    results_manager = ResultsManager()
    saved_path = results_manager.save_results(
        results=mock_results,
        output_dir="./results",
        experiment_name="test"
    )

    print(f"Results saved to: {saved_path}")
