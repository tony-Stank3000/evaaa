########################################################################################################
############################################## Figure b-2 ##############################################
########################################################################################################
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
from load_experiment_data import ExperimentDataLoader
from matplotlib.colors import LinearSegmentedColormap

class ExperimentAnalyzer:
    def __init__(self, base_dir: str):
        """
        Initialize the ExperimentAnalyzer with the base directory containing model folders.
        
        Args:
            base_dir (str): Path to the base directory containing model folders
        """
        self.base_dir = Path(base_dir)
        self.model_results: Dict[str, Dict[str, Dict[str, float]]] = {}  # Now stores both success rate and episode count
        self.experiment_types: List[str] = []
        
    def get_model_dirs(self) -> List[str]:
        """
        Get list of model directories in the base directory.
        
        Returns:
            List[str]: List of model directory names
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
    def calculate_success_rate(self, experiment_type: str, episodes_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate success rate for a specific experiment type.
        
        Success criteria:
        1. For two-resource-food, water, and goal-manipulation experiments:
            - Success if EpisodeEndType is ResourceConsumed AND
            - For food: FoodConsumed > 0
            - For water: WaterConsumed > 0
        2. For thermo experiment:
            - Success if EpisodeEndType is MaxStepReached
            - Failure otherwise
        3. For risk-taking experiment:
            - Success if EpisodeEndType is ResourceConsumed
            - Failure otherwise
        4. For Y-maze and multiGoalPlanning:
            - Success if EpisodeEndType is MaxStepReached
            - Failure otherwise
        5. For damage experiment:
            - Success if EpisodeEndType is ResourceConsumed
            - Failure otherwise
        
        Args:
            experiment_type (str): Type of experiment
            episodes_data (pd.DataFrame): Episodes data for the experiment
            
        Returns:
            Optional[float]: Success rate as a percentage, or None if calculation fails
        """
        if episodes_data is None or len(episodes_data) == 0:
            return None
            
        try:
            # Initialize counters
            total_episodes = len(episodes_data)
            successful_episodes = 0
            
            # Canonical experiment type keys
            food_keys = ['exp-two-resource-food', 'exp-goal-manipulation-WaterToFood', 'exp-predator']
            water_keys = ['exp-two-resource-water', 'exp-goal-manipulation-FoodToWater']
            thermo_key = 'exp-two-resource-thermo'
            risk_key = 'exp-riskTaking'
            ymaze_keys = ['exp-Ymaze', 'exp-multiGoalPlanning']
            damage_key = 'exp-damage'
            
            # Check each episode
            for _, episode in episodes_data.iterrows():
                if any(key in experiment_type for key in food_keys):
                    if (episode['EpisodeEndType'] == 'ResourceConsumed' and 
                        'FoodConsumed' in episode and episode['FoodConsumed'] > 0):
                        successful_episodes += 1
                elif any(key in experiment_type for key in water_keys):
                    if (episode['EpisodeEndType'] == 'ResourceConsumed' and 
                        'WaterConsumed' in episode and episode['WaterConsumed'] > 0):
                        successful_episodes += 1
                elif thermo_key in experiment_type:
                    if episode['EpisodeEndType'] == 'MaxStepReached':
                        successful_episodes += 1
                elif risk_key in experiment_type:
                    if episode['EpisodeEndType'] == 'ResourceConsumed':
                        successful_episodes += 1
                elif any(key in experiment_type for key in ymaze_keys):
                    if episode['EpisodeEndType'] == 'MaxStepReached':
                        successful_episodes += 1
                elif damage_key in experiment_type:
                    if episode['EpisodeEndType'] == 'ResourceConsumed':
                        successful_episodes += 1
            
            # Calculate success rate
            return (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating success rate for {experiment_type}: {str(e)}")
            return None
    
    def analyze_all_models(self, experiment_types: List[str]) -> None:
        """
        Analyze all models and calculate success rates for each experiment type.
        
        Args:
            experiment_types (List[str]): List of experiment types to analyze
        """
        self.experiment_types = experiment_types
        model_dirs = self.get_model_dirs()
        
        for model_dir in model_dirs:
            # Find the actual directory that matches the model name pattern
            matching_dirs = list(self.base_dir.glob(f"*{model_dir}*"))
            if not matching_dirs:
                print(f"Warning: No directory found for model {model_dir}")
                continue
                
            model_path = matching_dirs[0]
            print(f"Processing model directory: {model_path}")
            
            loader = ExperimentDataLoader(str(model_path))
            loader.load_all_data()
            
            model_results = {}
            for exp_type in experiment_types:
                episodes_data = loader.get_episodes_data(exp_type)
                if episodes_data is not None:
                    success_rate = self.calculate_success_rate(exp_type, episodes_data)
                    if success_rate is not None:
                        model_results[exp_type] = {
                            'success_rate': success_rate,
                            'episode_count': len(episodes_data)
                        }
            
            self.model_results[model_dir] = model_results
    
    def plot_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of success rates across models and experiment types.
        Maintains the exact order of models and experiments as specified in the lists.
        
        Args:
            save_path (Optional[str]): Path to save the plot. If None, plot is displayed.
        """
        if not self.model_results:
            print("No results to plot. Run analyze_all_models() first.")
            return
        
        if not self.experiment_types:
            print("No experiment types defined. Run analyze_all_models() first.")
            return
        
        # Get the ordered lists
        model_order = self.get_model_dirs()
        experiment_order = [exp.replace('exp-', '') for exp in self.experiment_types]
        
        # Create DataFrames for success rates and episode counts
        success_data = []
        count_data = []        
        cool_blue_map = LinearSegmentedColormap.from_list("cool_blue", ["#e5f1fb", "#9fc5e8", "#1f4e79"])

        
        for model in model_order:
            if model in self.model_results:
                for exp_type in self.experiment_types:
                    if exp_type in self.model_results[model]:
                        result = self.model_results[model][exp_type]
                        success_data.append({
                            'Model': model,
                            'Experiment': exp_type.replace('exp-', ''),
                            'Success Rate': result['success_rate'] / 100
                        })
                        count_data.append({
                            'Model': model,
                            'Experiment': exp_type.replace('exp-', ''),
                            'Episode Count': result['episode_count']
                        })
        
        # Create success rate heatmap
        plt.figure(figsize=(13, 12))
        
        # Success rate heatmap
        plt.subplot(2, 1, 1)
        success_df = pd.DataFrame(success_data)
        success_heatmap = success_df.pivot(index='Model', columns='Experiment', values='Success Rate')
        success_heatmap = success_heatmap.reindex(index=model_order, columns=experiment_order)
        
        sns.heatmap(success_heatmap, 
                annot=True, 
                fmt='.2f', 
                cmap=cool_blue_map,
                vmin=0, 
                vmax=1,
                annot_kws={"size": 16},
                cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rates by Model and Experiment')
        
        # Episode count heatmap
        plt.subplot(2, 1, 2)
        count_df = pd.DataFrame(count_data)
        count_heatmap = count_df.pivot(index='Model', columns='Experiment', values='Episode Count')
        count_heatmap = count_heatmap.reindex(index=model_order, columns=experiment_order)
        
        sns.heatmap(count_heatmap, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Episode Count'})
        plt.title('Episode Counts by Model and Experiment')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def main():
    base_dir = "Data_experiment/dreamer"

    model_types = [
        "level-4-2",
        "level-4-1",
        "level-3-2",
        "level-3-1",
        "level-2-2",
        "level-2-1",
        "level-1-2",
        "level-1-1",
    ]
    
    # List of experiment types to analyze
    experiment_types = [
        'exp-two-resource-food',
        'exp-two-resource-water',
        'exp-two-resource-thermo',
        'exp-damage',
        'exp-riskTaking',
        'exp-goal-manipulation-FoodToWater',
        'exp-goal-manipulation-WaterToFood',
        'exp-predator',
        'exp-Ymaze',
        'exp-multiGoalPlanning',
    ]
    
    print(f"Base directory: {base_dir}")
    print("\nAnalyzing models:")
    for model in model_types:
        print(f"- {model}")
    
    print("\nAnalyzing experiment types:")
    for exp_type in experiment_types:
        print(f"- {exp_type}")
    
    # Create analyzer
    analyzer = ExperimentAnalyzer(base_dir)
    
    # Override get_model_dirs to use our explicit list
    analyzer.get_model_dirs = lambda: model_types
    
    print("\nCalculating success rates...")
    analyzer.analyze_all_models(experiment_types)
    
    print("\nModel results:")
    for model, results in analyzer.model_results.items():
        print(f"\n{model}:")
        for exp_type, data in results.items():
            print(f"  {exp_type}:")
            print(f"    Success Rate: {data['success_rate']:.2f}%")
            print(f"    Episode Count: {data['episode_count']}")
    
    print("\nGenerating heatmap...")
    # analyzer.plot_heatmap(save_path=f"{base_dir}/experiment_performance.png")
    # print(f"Heatmap saved as: {os.path.abspath(f'{base_dir}/experiment_performance.png')}")
    analyzer.plot_heatmap(save_path="figure_b_2.png")
    print(f"Heatmap saved as: {os.path.abspath('figure_b_2.png')}")
    
    # Also display the plot
    plt.show()

if __name__ == "__main__":
    main() 
