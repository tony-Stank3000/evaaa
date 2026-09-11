import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

class ExperimentDataLoader:
    def __init__(self, experiment_dir: str, canonical_experiment_types: list = None):
        """
        Initialize the ExperimentDataLoader with the experiment directory path.
        Optionally, provide a list of canonical experiment types for robust mapping.
        
        Args:
            experiment_dir (str): Path to the experiment directory containing CSV files
            canonical_experiment_types (list): List of canonical experiment type names
        """
        self.experiment_dir = Path(experiment_dir)
        self.steps_data: Dict[str, pd.DataFrame] = {}
        self.episodes_data: Dict[str, pd.DataFrame] = {}
        self.canonical_experiment_types = canonical_experiment_types
        
    def _find_canonical_type(self, exp_type: str) -> str:
        """
        Find the canonical experiment type that is a substring of exp_type.
        If not found, return the original exp_type.
        """
        if self.canonical_experiment_types:
            for canonical in self.canonical_experiment_types:
                if canonical in exp_type:
                    return canonical
        return exp_type

    def load_all_data(self) -> None:
        """
        Load all CSV files from the experiment directory and organize them by experiment type.
        Uses substring matching to map to canonical experiment types if provided.
        """
        # Get all CSV files in the directory
        csv_files = list(self.experiment_dir.glob('*.csv'))
        
        for csv_file in csv_files:
            filename = csv_file.name
            if '_steps_' in filename:
                exp_type = filename.split('_steps_')[0]
                canonical_type = self._find_canonical_type(exp_type)
                self.steps_data[canonical_type] = pd.read_csv(csv_file)
            elif '_episodes_' in filename:
                exp_type = filename.split('_episodes_')[0]
                canonical_type = self._find_canonical_type(exp_type)
                self.episodes_data[canonical_type] = pd.read_csv(csv_file)
    
    def get_steps_data(self, experiment_type: str) -> pd.DataFrame:
        """
        Get steps data for a specific experiment type.
        Uses substring matching for robustness.
        """
        for key in self.steps_data:
            if experiment_type in key or key in experiment_type:
                return self.steps_data[key]
        return None
    
    def get_episodes_data(self, experiment_type: str) -> pd.DataFrame:
        """
        Get episodes data for a specific experiment type.
        Uses substring matching for robustness.
        """
        for key in self.episodes_data:
            if experiment_type in key or key in experiment_type:
                return self.episodes_data[key]
        return None
    
    def get_all_experiment_types(self) -> List[str]:
        """
        Get a list of all experiment types available in the data.
        
        Returns:
            List[str]: List of experiment types
        """
        return list(self.steps_data.keys())

def main():
    experiment_dir = "Data_experiment/dreamer"
    canonical_experiment_types = [
        'exp-two-resource-food',
        'exp-two-resource-water',
        'exp-two-resource-thermo',
        'exp-damage',
        'exp-riskTaking',
        'exp-goal-manipulation-food',
        'exp-goal-manipulation-water',
        'predator',
        'exp-Ymaze',
        'exp-multiGoalPlanning',
    ]
    loader = ExperimentDataLoader(experiment_dir, canonical_experiment_types)
    loader.load_all_data()
    
    # Print available experiment types
    print("Available experiment types:")
    for exp_type in loader.get_all_experiment_types():
        print(f"- {exp_type}")
    
    # Example of accessing data
    ymaze_steps = loader.get_steps_data('exp-Ymaze')
    if ymaze_steps is not None:
        print("\nYmaze steps data shape:", ymaze_steps.shape)
        print("\nYmaze steps data columns:", ymaze_steps.columns.tolist())
        print("\nFirst few rows of Ymaze steps data:")
        print(ymaze_steps.head())

if __name__ == "__main__":
    main() 