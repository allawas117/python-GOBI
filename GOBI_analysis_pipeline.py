import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from itertools import combinations
from tqdm import tqdm
import scipy.io
from joblib import Parallel, delayed
import time
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CausalInference:
    def __init__(self):
        self._data = None
        self._rds_data = None
        self._params = {
            'thres_R': 0.05,
            'thres_S': 0.9,
            'thres_TRS': 0.9,
            'thres_noise': 0,
            'type_self': np.nan,
            'win_avg': 14,
            'method': 2,
            'num_fourier': 8,
            'window_size_ori': 20,
            'overlapping_ratio': 0.5,
            'time_interval': 1
        }
        self._results = {}
        self._dimension = 1
        self._visualizer = CausalVisualizer(self)
        self._data_processor = DataProcessor()

    @property
    def dimension(self):
        return self._dimension
    
    @dimension.setter
    def dimension(self, value):
        if not isinstance(value, int) or value < 1 or value > 3:
            raise ValueError("Dimension must be an integer between 1 and 3")
        self._dimension = value

    @property
    def params(self):
        return self._params.copy()  # Return a copy to prevent direct modification

    def set_params(self, **kwargs):
        """Protected method to update parameters with validation"""
        for key, value in kwargs.items():
            if key in self._params:
                self._params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def tune_parameters(self, method='interactive'):
        """
        Tune parameters for the model.
        
        Parameters:
        -----------
        method : str
            'interactive': Interactively prompt for parameters
            'auto': Perform automated hyperparameter tuning
        
        Returns:
        --------
        dict
            Selected parameters
        """
        if method == 'interactive':
            print("=== Interactive Parameter Selection ===")
            print("Current parameters:")
            for param, value in self._params.items():
                print(f"  {param}: {value}")
            
            print("\nEnter new parameter values (leave blank to keep current value):")
            new_params = {}
            
            for param, value in self._params.items():
                # Customize prompt based on parameter type
                if param == 'type_self':
                    prompt = f"Enter {param} (current: {value}, use 'nan' for NaN): "
                    new_val = input(prompt)
                    if new_val:
                        if new_val.lower() == 'nan':
                            new_params[param] = np.nan
                        else:
                            try:
                                new_params[param] = float(new_val)
                            except ValueError:
                                print(f"Invalid value for {param}, keeping current value.")
                else:
                    prompt = f"Enter {param} (current: {value}): "
                    new_val = input(prompt)
                    if new_val:
                        try:
                            # Convert to appropriate type
                            if isinstance(value, int):
                                new_params[param] = int(new_val)
                            elif isinstance(value, float):
                                new_params[param] = float(new_val)
                            else:
                                new_params[param] = type(value)(new_val)
                        except ValueError:
                            print(f"Invalid value for {param}, keeping current value.")
            
            if new_params:
                self.set_params(**new_params)
                print("Parameters updated successfully.")
            else:
                print("No parameters were changed.")
            
            return new_params
        
        elif method == 'auto':
            print("=== Automated Hyperparameter Tuning ===")
            print("Select parameters to tune:")
            
            # Offer common parameters to tune
            available_params = {
                'thres_R': [0.03, 0.05, 0.07, 0.09],
                'thres_S': [0.8, 0.85, 0.9, 0.95],
                'thres_TRS': [0.8, 0.85, 0.9, 0.95],
                'thres_noise': [0, 0.01, 0.02],
                'win_avg': [7, 14, 21]
            }
            
            # Let user select parameters to tune
            params_to_tune = {}
            for param, default_values in available_params.items():
                tune_this = input(f"Tune {param}? (y/n, default: y): ")
                if tune_this.lower() != 'n':
                    custom_values = input(f"Enter values to try for {param} separated by commas (default: {default_values}): ")
                    if custom_values:
                        try:
                            # Convert to appropriate type
                            current_value = self._params[param]
                            if isinstance(current_value, int):
                                params_to_tune[param] = [int(val.strip()) for val in custom_values.split(',')]
                            elif isinstance(current_value, float):
                                params_to_tune[param] = [float(val.strip()) for val in custom_values.split(',')]
                            else:
                                params_to_tune[param] = [type(current_value)(val.strip()) for val in custom_values.split(',')]
                        except ValueError:
                            print(f"Invalid values for {param}, using default values.")
                            params_to_tune[param] = default_values
                    else:
                        params_to_tune[param] = default_values
            
            if not params_to_tune:
                print("No parameters selected for tuning.")
                return {}
            
            # Configure tuning options
            search_type = input("Search type (grid/random, default: random): ")
            search_type = search_type.lower() if search_type else "random"
            
            if search_type == "random":
                n_iterations = input("Number of random iterations (default: 10): ")
                n_iterations = int(n_iterations) if n_iterations and n_iterations.isdigit() else 10
            else:
                n_iterations = 0  # Use grid search
            
            cv_folds = input("Number of cross-validation folds (default: 3): ")
            cv_folds = int(cv_folds) if cv_folds and cv_folds.isdigit() else 3
            
            scoring_method = input("Scoring method (trs_variance/trs_mean/connectivity, default: trs_variance): ")
            scoring_method = scoring_method if scoring_method in ['trs_variance', 'trs_mean', 'connectivity'] else 'trs_variance'
            
            # Run hyperparameter tuning
            best_params, all_results = self.add_hyperparameter_tuning(
                parameter_ranges=params_to_tune,
                scoring_method=scoring_method,
                n_iterations=n_iterations,
                cv_folds=cv_folds
            )
            
            # Ask if user wants to visualize results
            visualize = input("Visualize tuning results? (y/n, default: y): ")
            if visualize.lower() != 'n':
                self._visualizer.plot_hyperparameter_tuning_results(all_results)
            
            return best_params
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def add_hyperparameter_tuning(self, parameter_ranges, scoring_method='trs_variance', n_iterations=10, cv_folds=5):
        """
        Perform hyperparameter tuning on the causal inference model.
        
        Parameters:
        -----------
        parameter_ranges : dict
            Dictionary with parameter names as keys and lists of values to try
            Example: {'thres_R': [0.03, 0.05, 0.07], 'thres_S': [0.8, 0.85, 0.9]}
        scoring_method : str
            Method to score parameter combinations ('trs_variance', 'trs_mean', 'connectivity')
        n_iterations : int
            Number of random iterations if using random search
        cv_folds : int
            Number of cross-validation folds
        
        Returns:
        --------
        best_params : dict
            Best parameter combination found
        all_results : list
            List of all parameter combinations and their scores
        """
        from sklearn.model_selection import KFold
        import itertools
        import random
        
        if 'y_total' not in self._data:
            print("Please run preprocessing first.")
            return {}, []
        
        all_results = []
        
        # Store original parameters
        original_params = self._params.copy()
        
        # Create data splits for cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        data_indices = np.arange(self._data['num_data'])
        
        # Decide whether to do grid search or random search
        if n_iterations > 0 and n_iterations < np.prod([len(values) for values in parameter_ranges.values()]):
            # Random search
            param_combinations = []
            for _ in range(n_iterations):
                combo = {k: random.choice(v) for k, v in parameter_ranges.items()}
                param_combinations.append(combo)
        else:
            # Grid search
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            param_combinations = [dict(zip(param_names, combo)) 
                                for combo in itertools.product(*param_values)]
        
        # Function to evaluate a single parameter combination
        def evaluate_params(params_to_test, train_indices, test_indices):
            # Update parameters
            self._params.update(params_to_test)
            
            # Create subset of data for training
            original_y_total = self._data['y_total'].copy()
            self._data['y_total'] = original_y_total[train_indices]
            self._data['num_data'] = len(train_indices)
            
            # Run analysis on training data
            self.run_analysis()
            
            # Score on test data
            self._data['y_total'] = original_y_total[test_indices]
            self._data['num_data'] = len(test_indices)
            
            # Calculate test score based on the scoring method
            # Current metrics for scoring methods need more evidence!!!
            if scoring_method == 'trs_variance':
                # Lower variance in TRS scores is better
                TRS_total = self._results['TRS_total']
                score = -np.nanvar(TRS_total)  # Negative because we want to maximize score
            elif scoring_method == 'trs_mean':
                # Higher mean TRS is better (closer to 1)
                TRS_total = self._results['TRS_total']
                score = np.nanmean(TRS_total)
            elif scoring_method == 'connectivity':
                # More significant connections is better
                TRS_total = self._results['TRS_total']
                significant_connections = np.sum(TRS_total > self._params['thres_TRS'])
                score = significant_connections
            else:
                raise ValueError(f"Unknown scoring method: {scoring_method}")
            
            # Restore original data
            self._data['y_total'] = original_y_total
            self._data['num_data'] = len(original_y_total)
            
            return score
        
        print(f"Starting hyperparameter tuning with {len(param_combinations)} parameter combinations...")
        
        # Evaluate each parameter combination with cross-validation
        for i, params_to_test in enumerate(param_combinations):
            cv_scores = []
            
            for train_idx, test_idx in kf.split(data_indices):
                score = evaluate_params(params_to_test, train_idx, test_idx)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            all_results.append({
                'params': params_to_test,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            print(f"Progress: {i+1}/{len(param_combinations)}, Score: {mean_score:.4f}±{std_score:.4f}, Params: {params_to_test}")
        
        # Find best parameters
        all_results.sort(key=lambda x: x['mean_score'], reverse=True)
        best_params = all_results[0]['params']
        
        # Restore original parameters
        self._params = original_params
        
        print(f"Hyperparameter tuning complete. Best parameters: {best_params}")
        print(f"Best score: {all_results[0]['mean_score']:.4f}±{all_results[0]['std_score']:.4f}")
        
        # Offer to update parameters with best found
        update = input("Update model with best parameters? (y/n): ")
        if update.lower() == 'y':
            self._params.update(best_params)
            print("Parameters updated.")
        
        return best_params, all_results

    def load_data(self, file_path):
        self._data_processor.load_data(file_path)
        self._data = self._data_processor.data
        self._dimension = self.detect_dimension(self._data['num_component'])
        print(f"Automatically detected dimension: {self._dimension}")

    def preprocess(self):
        self._data_processor.preprocess(self._params)
        self._data = self._data_processor.data

    def detect_dimension(self, n_components):
        """
        Automatically detect the dimension based on number of components.
        For n components, the dimension is typically n-1 (for meaningful interactions).
        Maximum supported dimension is 3. 
        Implementation for higher dimensions may be included in the future.
        Currently, some higher dimension analysis using GOBI may cause errors.
        """
        if n_components <= 1:
            return 1  # Default to dimension 1 if only one component
        elif n_components >= 4:
            return 3  # Cap at dimension 3 for practical reasons
        else:
            return n_components - 1

    def run_analysis(self):
        if 'y_total' not in self._data:
            print("Please run preprocessing first.")
            return

        print(f"Starting analysis for dimension {self._dimension}...")
        start_time = time.time()

        thres_R = self._params['thres_R']
        thres_S = self._params['thres_S']
        thres_TRS = self._params['thres_TRS']
        thres_noise = self._params['thres_noise']
        type_self = self._params['type_self']
        time_interval = self._params['time_interval']

        # Generate component list using the generalized function
        component_list = self.generate_component_list_dimN(
            self._data['num_component'],
            self._dimension,
            self._params['type_self']
        )

        # Number of regulation types is 2^dimension
        num_type = 2**self._dimension

        print(f"Number of components: {self._data['num_component']}")
        print(f"Number of regulation types: {num_type}")
        print(f"Component list shape: {component_list.shape}")

        # Compute scores using the generalized compute_scores_parallel function
        S_total_list, R_total_list = self.compute_scores_parallel(component_list, num_type)

        # Process scores
        S_processed = self.S_threshold(S_total_list, self._params['thres_S'])
        R_processed = self.R_threshold(R_total_list, self._params['thres_R'])

        R_sum = np.sum(R_processed, axis=2)
        SR_sum = np.sum(S_processed * R_processed, axis=2)

        TRS_total = np.where(R_sum == 0, np.nan, SR_sum / R_sum)

        self._results = {
            'TRS_total': TRS_total,
            'component_list': component_list,
            'S_total_list': S_total_list,  # Store the raw scores
            'R_total_list': R_total_list,  # Store the raw scores
            'dimension': self._dimension
        }

        print(f"Analysis completed in {time.time() - start_time:.2f} seconds.")

    def generate_component_list_dimN(self, num_component, dimension, type_self):
        """
        Generate component list for any dimension N.
        Format: [cause1, cause2, ..., causeN, target]
        """
        component_list = []

        # Function to generate combinations of causes
        def generate_cause_combinations(components, dimension):
            return list(combinations(components, dimension))

        # Generate all cause combinations
        component_indices = list(range(1, num_component + 1))
        cause_combinations = generate_cause_combinations(component_indices, dimension)

        # For each combination, iterate through possible targets
        for causes in cause_combinations:
            for target in range(1, num_component + 1):
                # Only include if target is not in the causes OR type_self is not NaN
                if target not in causes or not np.isnan(type_self):
                    component_list.append(list(causes) + [target])

        return np.array(component_list)

    def compute_scores_parallel(self, component_list, num_type):
        """Compute scores for any dimension analysis in parallel."""
        y_total = self._data['y_total']
        t = self._data['t'].flatten()
        num_data = len(y_total)
        num_pair = len(component_list)

        S_total_list = np.zeros((num_pair, num_type, num_data))
        R_total_list = np.zeros((num_pair, num_type, num_data))

        def process_data(i):
            y_target = y_total[i]
            t_target = t.flatten()

            S_total = np.zeros((num_pair, num_type))
            R_total = np.zeros((num_pair, num_type))

            for j, combo in enumerate(component_list):
                # Extract causes and target
                # Last element is the target, all others are causes
                cause_indices = combo[:-1]
                target_index = combo[-1]

                # Get cause and target variables
                X_list = [y_target[:, idx-1] for idx in cause_indices]
                Y = y_target[:, target_index-1]

                # Use the generalized RDS function
                score_list, t_1, t_2 = self.RDS_dimN(X_list, Y, t_target, self._params['time_interval'])

                # Calculate S and R values as before
                for k in range(num_type):
                    score = score_list[:, :, k]
                    loca_plus = np.where(score > self._params['thres_noise'])
                    loca_minus = np.where(score < -self._params['thres_noise'])

                    if len(loca_plus[0]) == 0 and len(loca_minus[0]) == 0:
                        s = 1
                    else:
                        s = (np.sum(score[loca_plus]) + np.sum(score[loca_minus])) / (np.abs(np.sum(score[loca_plus])) + np.abs(np.sum(score[loca_minus])))

                    r = (len(loca_plus[0]) + len(loca_minus[0])) / (len(t_1) * len(t_2) / 2)
                    S_total[j, k] = s
                    R_total[j, k] = r

            return S_total, R_total

        results = Parallel(n_jobs=-1)(delayed(process_data)(i) for i in range(num_data))

        for i, (S_total, R_total) in enumerate(results):
            S_total_list[:, :, i] = S_total
            R_total_list[:, :, i] = R_total

        return S_total_list, R_total_list

    def RDS_dimN(self, X_list, Y, t, time_interval):
        """
        Generalized Regulation Detection Scoring function for any dimension N.
        Based on the MATLAB implementation. 
        Potential for numerical differences due to MATLAB's matrix operations and toolbox dependencies.
        """
        # Calculate the gradient of target variable
        f = np.gradient(Y, time_interval, edge_order=1)

        # Create meshgrids and differences for each cause
        X_diffs = []
        for X in X_list:
            X_mesh, X_mesh_transpose = np.meshgrid(X, X)
            X_diffs.append(X_mesh_transpose - X_mesh)

        # Create meshgrid for the target gradient
        f_mesh, f_mesh_transpose = np.meshgrid(f, f)
        f_diff = f_mesh_transpose - f_mesh

        # Number of regulation types: 2^dimension
        num_types = 2**self._dimension

        # Initialize scores array with zeros
        score_list = np.zeros((len(t), len(t), num_types))
        
        # Calculate base score (X_diffs product * f_diff)
        base_score = f_diff
        for X_diff in X_diffs:
            base_score = base_score * X_diff
        
        # For each regulation type
        for i in range(num_types):
            # Convert type index to binary pattern
            pattern = [(i >> j) & 1 for j in range(len(X_list))]
            
            # Create conditions array for this regulation type
            conditions = []
            sign = 1  # Track the final sign for this regulation type
            
            for idx, (bit, X_diff) in enumerate(zip(pattern, X_diffs)):
                if bit == 0:  # Positive regulation
                    conditions.append(X_diff >= 0)
                else:  # Negative regulation
                    conditions.append(X_diff < 0)
                    sign *= -1  # Flip sign for each negative regulation (follows GOBI MATLAB logic)
            
            # Combine all conditions
            combined_condition = conditions[0]
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
                
            # Apply score with proper sign (matching MATLAB implementation)
            score_list[:, :, i] = np.where(combined_condition, sign * base_score, 0)

        t_1, t_2 = np.meshgrid(t, t)
        return score_list, t_1, t_2

    def S_threshold(self, S, thres):
        return np.where(np.abs(S) < thres, 1, 0)

    def R_threshold(self, R, thres):
        return np.where(R > thres, 1, 0)

    def save_results(self, filename):
        np.save(filename, self._results)
        print(f"Results saved to {filename}")

    def load_results(self, filename):
        self._results = np.load(filename, allow_pickle=True).item()
        print(f"Results loaded from {filename}")

class DataProcessor:
    """Handles all data loading and preprocessing operations"""
    def __init__(self):
        self._data = None

    @property
    def data(self):
        return self._data

    def load_data(self, file_path):
        print("Loading data...")
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'mat':
            # Load .mat file
            mat_data = scipy.io.loadmat(file_path)

            # Find t and y arrays in the .mat file
            t_key = next((key for key in mat_data.keys() if key.lower() == 't'), None)
            y_key = next((key for key in mat_data.keys() if key.lower() == 'y'), None)

            if t_key is None or y_key is None:
                raise ValueError("Could not find 't' and 'y' arrays in the .mat file")

            t = mat_data[t_key].flatten()
            y = mat_data[y_key]

            print(f"Loaded data: t from '{t_key}', y from '{y_key}'")
        elif file_extension == 'csv':
            # Load .csv file
            data = pd.read_csv(file_path)
            t = data.iloc[:, 0].values
            y = data.iloc[:, 1:].values
        elif file_extension in ['xlsx', 'xls']:
            # Load .xlsx or .xls file
            data = pd.read_excel(file_path)
            t = data.iloc[:, 0].values
            y = data.iloc[:, 1:].values
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Ensure t is a 1D array and y is a 2D array
        t = np.array(t).flatten()
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Store data
        self._data = {
            'y': y,
            't': t,
            'num_component': y.shape[1],
            'num_columns': y.shape[1]
        }

        print(f"Data loaded successfully. Shape of y: {y.shape}, Shape of t: {t.shape}")

    def preprocess(self, params):
        print("Starting preprocessing...")
        start_time = time.time()

        y = self._data['y']
        t = self._data['t']
        num_columns = self._data['num_columns']

        # Moving average
        win_avg = params['win_avg']
        y_movavg = np.apply_along_axis(lambda m: np.convolve(m, np.ones(win_avg), 'same') / win_avg, axis=0, arr=y)

        # Interpolation parameters
        method = params['method']
        num_fourier = params['num_fourier']
        window_size_ori = params['window_size_ori']
        overlapping_ratio = params['overlapping_ratio']
        time_interval = params['time_interval']

        window_size = int(window_size_ori / time_interval)
        window_move = int(np.ceil(window_size_ori * (1 - overlapping_ratio)) / time_interval)

        # Interpolate data
        t_original = np.arange(len(t))
        t_fit = np.linspace(t_original[0], t_original[-1], int(len(t_original) / time_interval))
        y_fit = np.zeros((len(t_fit), num_columns))

        for i in range(num_columns):
            if method == 1:
                interp_func = interp1d(t_original, y_movavg[:, i], kind='linear')
            elif method == 2:
                interp_func = interp1d(t_original, y_movavg[:, i], kind='cubic')
            elif method == 3:
                interp_func = interp1d(t_original, y_movavg[:, i], kind='linear')
                y_tmp = interp_func(t_fit)

                def fourier_func(x, *params):
                    y = np.zeros_like(x)
                    for i in range(0, len(params) - 1, 2):
                        y += params[i] * np.cos(params[i+1] * x) + params[i+2] * np.sin(params[i+1] * x)
                    return y + params[-1]

                initial_params = [1] * (2 * num_fourier + 1)
                params, _ = curve_fit(fourier_func, t_fit, y_tmp, p0=initial_params)
                y_fit[:, i] = fourier_func(t_fit, *params)
                continue

            y_fit[:, i] = interp_func(t_fit)

        # Cut and normalize data
        y_total = []
        for start in range(0, len(y_fit) - window_size + 1, window_move):
            y_tmp = y_fit[start:start + window_size, :]
            y_tmp = (y_tmp - np.min(y_tmp, axis=0)) / (np.max(y_tmp, axis=0) - np.min(y_tmp, axis=0))
            y_total.append(y_tmp)

        self._data['y_total'] = np.array(y_total)
        self._data['t'] = t_fit[:window_size] / t_fit[window_size - 1]
        self._data['time_interval'] = time_interval
        self._data['num_data'] = len(y_total)

        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

class CausalVisualizer:
    """Handles all visualization operations"""
    def __init__(self, causal_inference):
        self._ci = causal_inference
        
    def plot_trs_heatmap(self):
        if 'TRS_total' not in self._ci._results:
            print("Please run the analysis first.")
            return

        TRS_total = self._ci._results['TRS_total']
        component_list = self._ci._results['component_list']

        plt.figure(figsize=(10, 8))
        plt.imshow(TRS_total, aspect='auto', cmap='Blues_r', vmin=0, vmax=1)
        plt.colorbar()

        for i in range(TRS_total.shape[0] + 1):
            plt.axhline(y=i - 0.5, color='black', linewidth=1)
        for i in range(TRS_total.shape[1] + 1):
            plt.axvline(x=i - 0.5, color='black', linewidth=1)

        plt.title(f'Heatmap of TRS (Dimension {self._ci._dimension} Analysis)')
        plt.xlabel('Regulation Type')
        plt.ylabel('Component Combinations')

        # Create labels based on dimension
        if self._ci._dimension == 1:
            y_labels = [f"{combo[0]} -> {combo[1]}" for combo in component_list]
            x_labels = ['Positive', 'Negative']
        elif self._ci._dimension == 2:
            y_labels = [f"({combo[0]},{combo[1]}) -> {combo[2]}" for combo in component_list]
            x_labels = ['++', '+-', '-+', '--']  # Four regulation types for dimension 2

        plt.yticks(range(len(y_labels)), y_labels)
        plt.xticks(range(len(x_labels)), x_labels)

        plt.tight_layout()
        plt.show()
        
    def plot_time_series(self):
        if 'y_total' not in self._ci._data:
            print("Please run preprocessing first.")
            return

        y_total = self._ci._data['y_total']  # Windowed data
        t = self._ci._data['t']  # Time points within a window
        num_windows = len(y_total)

        # Concatenate windowed data for plotting
        y_fit = np.concatenate(y_total, axis=0)
        t_fit = np.tile(t, num_windows)  # Repeat time points for each window
        t_fit = t_fit + np.repeat(np.arange(num_windows) * (len(t) - int(len(t) * self._ci._params['overlapping_ratio'])), len(t))

        plt.figure(figsize=(12, 6))
        plt.plot(t_fit, y_fit, 'b-', linewidth=1, label='Windowed Data')  # Plot windowed data
        plt.title('Time Series: Windowed Data')  # Update title
        plt.xlabel('Time')
        plt.ylabel('Value')  # Use 'Value' for windowed data
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_rds_scores_time_windows(self, component_indices=None, regulation_types=None, max_windows=None):
        """
        Plot RDS across time windows for selected components and regulation types.

        Parameters:
        -----------
        component_indices : list or None
            Indices of components to plot. If None, plot all components.
        regulation_types : list or None
            Indices of regulation types to plot. If None, plot all types.
        max_windows : int or None
            Maximum number of time windows to plot. If None, plot all windows.
        """
        if 'TRS_total' not in self._ci._results:
            print("Please run the analysis first.")
            return

        S_total_list = self._ci._results.get('S_total_list', None)
        R_total_list = self._ci._results.get('R_total_list', None)

        if S_total_list is None or R_total_list is None:
            print("S_total_list or R_total_list not found in results. Ensure you're storing these in the results dictionary.")
            return

        component_list = self._ci._results['component_list']
        num_windows = S_total_list.shape[2]

        # If no specific components selected, plot all
        if component_indices is None:
            component_indices = range(len(component_list))

        # If no specific regulation types selected, plot all
        if regulation_types is None:
            regulation_types = range(2**self._ci._dimension)

        # Limit the number of windows if specified
        if max_windows is not None and max_windows < num_windows:
            windows_to_plot = np.linspace(0, num_windows-1, max_windows, dtype=int)
        else:
            windows_to_plot = range(num_windows)

        # Create labels for regulation types
        if self._ci._dimension == 1:
            reg_type_labels = ['Positive', 'Negative']
        elif self._ci._dimension == 2:
            reg_type_labels = ['++', '+-', '-+', '--']
        elif self._ci._dimension == 3:
            reg_type_labels = ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']
        else:
            reg_type_labels = [f'Type {i}' for i in range(2**self._ci._dimension)]

        # Create subplot grid
        n_components = len(component_indices)
        n_reg_types = len(regulation_types)

        fig, axes = plt.subplots(n_components, n_reg_types, figsize=(4*n_reg_types, 3*n_components),
                                sharex=True, sharey=True)

        # Adjust axes for single component or regulation type
        if n_components == 1 and n_reg_types == 1:
            axes = np.array([[axes]])
        elif n_components == 1:
            axes = axes.reshape(1, -1)
        elif n_reg_types == 1:
            axes = axes.reshape(-1, 1)

        # Plot each component and regulation type
        for i, comp_idx in enumerate(component_indices):
            combo = component_list[comp_idx]
            if self._ci._dimension == 1:
                combo_label = f"{combo[0]} -> {combo[1]}"
            else:
                cause_str = ','.join(str(c) for c in combo[:-1])
                combo_label = f"({cause_str}) -> {combo[-1]}"

            for j, reg_type in enumerate(regulation_types):
                ax = axes[i, j]

                # Extract S and R values for this component and regulation type across windows
                S_values = S_total_list[comp_idx, reg_type, windows_to_plot]
                R_values = R_total_list[comp_idx, reg_type, windows_to_plot]

                # Plot S values
                ax.plot(windows_to_plot, S_values, 'b-', label='S Score')
                ax.set_ylim(-1.1, 1.1)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

                # Plot R values on a second y-axis
                ax2 = ax.twinx()
                ax2.plot(windows_to_plot, R_values, 'r-', label='R Score')
                ax2.set_ylim(0, 1.1)

                # Set titles and labels
                if i == 0:
                    ax.set_title(f'Type: {reg_type_labels[reg_type]}')
                if j == 0:
                    ax.set_ylabel(f'S Score\n{combo_label}')
                if j == n_reg_types - 1:
                    ax2.set_ylabel('R Score')
                if i == n_components - 1:
                    ax.set_xlabel('Time Window')

        # Add a legend
        lines1, labels1 = axes[0, 0].get_legend_handles_labels()
        lines2, labels2 = axes[0, 0].get_figure().axes[1].get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.suptitle(f'RDS Scores Across Time Windows (Dimension {self._ci._dimension})', fontsize=16)
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        plt.show()

    def plot_rds_component_comparison(self, component_indices=None, max_windows=None):
        """
        Plot S-scores for different regulation types together for each component.

        Parameters:
        -----------
        component_indices : list or None
            Indices of components to plot. If None, plot all components.
        max_windows : int or None
            Maximum number of time windows to plot. If None, plot all windows.
        """
        if 'S_total_list' not in self._ci._results:
            print("Please run the analysis first.")
            return

        S_total_list = self._ci._results['S_total_list']
        component_list = self._ci._results['component_list']
        num_windows = S_total_list.shape[2]
        num_types = 2**self._ci._dimension

        # If no specific components selected, plot all
        if component_indices is None:
            component_indices = range(len(component_list))

        # Limit the number of windows if specified
        if max_windows is not None and max_windows < num_windows:
            windows_to_plot = np.linspace(0, num_windows-1, max_windows, dtype=int)
        else:
            windows_to_plot = range(num_windows)

        # Create labels for regulation types
        if self._ci._dimension == 1:
            reg_type_labels = ['Positive', 'Negative']
            colors = ['blue', 'red']
        elif self._ci._dimension == 2:
            reg_type_labels = ['++', '+-', '-+', '--']
            colors = ['blue', 'green', 'orange', 'red']
        else:
            reg_type_labels = [f'Type {i}' for i in range(num_types)]
            colors = plt.cm.tab10(np.linspace(0, 1, num_types))

        # Create subplots, one for each component
        n_components = len(component_indices)
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 4*n_components), sharex=True)

        # Handle single component case
        if n_components == 1:
            axes = [axes]

        # Plot each component
        for i, comp_idx in enumerate(component_indices):
            ax = axes[i]
            combo = component_list[comp_idx]

            # Create component label
            if self._ci._dimension == 1:
                combo_label = f"{combo[0]} -> {combo[1]}"
            else:
                cause_str = ','.join(str(c) for c in combo[:-1])
                combo_label = f"({cause_str}) -> {combo[-1]}"

            # Plot each regulation type
            for reg_type in range(num_types):
                # Extract S values for this component and regulation type
                S_values = S_total_list[comp_idx, reg_type, windows_to_plot]

                # Plot S values
                ax.plot(windows_to_plot, S_values, color=colors[reg_type],
                        label=reg_type_labels[reg_type], linewidth=2)

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

            # Set labels and title
            ax.set_ylabel('S-Score')
            ax.set_title(f'Component: {combo_label}')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        # Set x-axis label for bottom subplot
        axes[-1].set_xlabel('Time Window')

        plt.tight_layout()
        plt.suptitle(f'RDS S-Scores by Regulation Type (Dimension {self._ci._dimension})', fontsize=16, y=1.02)
        plt.show()

    def plot_time_window_segmentation(self, component_indices=None, max_windows=5):
        """
        Visualize the time window segmentation of the data.

        Parameters:
        -----------
        component_indices : list or None
            Indices of components to plot. If None, plot first component.
        max_windows : int
            Maximum number of time windows to plot.
        """
        if 'y_total' not in self._ci._data:
            print("Please run preprocessing first.")
            return

        y_total = self._ci._data['y_total']
        num_windows = len(y_total)

        # If no specific components selected, plot the first component
        if component_indices is None:
            component_indices = [0]

        # Select a subset of windows to plot
        if max_windows > num_windows:
            max_windows = num_windows

        window_indices = np.linspace(0, num_windows-1, max_windows, dtype=int)

        # Create a grid of subplots
        fig, axes = plt.subplots(len(component_indices), max_windows,
                                figsize=(4*max_windows, 3*len(component_indices)),
                                sharex=True, sharey=True)

        # Adjust axes for single component or window
        if len(component_indices) == 1 and max_windows == 1:
            axes = np.array([[axes]])
        elif len(component_indices) == 1:
            axes = axes.reshape(1, -1)
        elif max_windows == 1:
            axes = axes.reshape(-1, 1)

        # Plot each component and window
        for i, comp_idx in enumerate(component_indices):
            for j, win_idx in enumerate(window_indices):
                ax = axes[i, j]

                # Extract data for this component and window
                window_data = y_total[win_idx][:, comp_idx]

                # Plot the data
                ax.plot(window_data)

                # Set titles and labels
                if i == 0:
                    ax.set_title(f'Window {win_idx+1}')
                if j == 0:
                    ax.set_ylabel(f'Component {comp_idx+1}')
                if i == len(component_indices) - 1:
                    ax.set_xlabel('Time')

        plt.tight_layout()
        plt.suptitle('Time Window Segmentation Visualization', fontsize=16)
        plt.show()

    def plot_s_score_heatmap(self, component_indices=None):
        """
        Plot S-scores as a heatmap for each component across time windows.

        Parameters:
        -----------
        component_indices : list or None
            Indices of components to plot. If None, plot all components.
        """
        if 'S_total_list' not in self._ci._results:
            print("Please run the analysis first.")
            return

        S_total_list = self._ci._results['S_total_list']
        component_list = self._ci._results['component_list']

        # If no specific components selected, plot all
        if component_indices is None:
            component_indices = range(len(component_list))

        num_windows = S_total_list.shape[2]
        num_types = 2**self._ci._dimension

        # Create labels for regulation types
        if self._ci._dimension == 1:
            reg_type_labels = ['Positive', 'Negative']
        elif self._ci._dimension == 2:
            reg_type_labels = ['++', '+-', '-+', '--']
        else:
            reg_type_labels = [f'Type {i}' for i in range(num_types)]

        # Create subplots, one for each component
        n_components = len(component_indices)
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))

        # Handle single component case
        if n_components == 1:
            axes = [axes]

        # Plot each component
        for i, comp_idx in enumerate(component_indices):
            ax = axes[i]
            combo = component_list[comp_idx]

            # Create component label
            if self._ci._dimension == 1:
                combo_label = f"{combo[0]} -> {combo[1]}"
            else:
                cause_str = ','.join(str(c) for c in combo[:-1])
                combo_label = f"({cause_str}) -> {combo[-1]}"

            # Extract S values for all regulation types
            data = S_total_list[comp_idx, :, :]

            # Plot as heatmap
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                          interpolation='none', extent=[0, num_windows, num_types, 0])

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

            # Set y-ticks to regulation types
            ax.set_yticks(np.arange(0.5, num_types))
            ax.set_yticklabels(reg_type_labels)

            # Set title
            ax.set_title(f'Component: {combo_label}')

            # Set labels
            ax.set_ylabel('Regulation Type')
            if i == n_components - 1:
                ax.set_xlabel('Time Window')

        plt.tight_layout()
        plt.suptitle(f'S-Score Heatmap Across Time Windows (Dimension {self._ci._dimension})', fontsize=16, y=1.02)
        plt.show()

    def plot_hyperparameter_tuning_results(self, tuning_results):
        """
        Plot the results of hyperparameter tuning to understand parameter importance.
        
        Parameters:
        -----------
        tuning_results : list
            Output from add_hyperparameter_tuning method
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {**r['params'], 'score': r['mean_score'], 'std': r['std_score']} 
            for r in tuning_results
        ])
        
        # Sort by score
        results_df = results_df.sort_values('score', ascending=False)
        
        # Plot top parameter combinations
        plt.figure(figsize=(10, 6))
        top_n = min(10, len(results_df))
        plt.errorbar(
            range(top_n),
            results_df['score'].iloc[:top_n],
            yerr=results_df['std'].iloc[:top_n],
            fmt='o'
        )
        plt.xticks(
            range(top_n),
            [str({k: v for k, v in row.items() if k not in ['score', 'std']}) 
            for _, row in results_df.iloc[:top_n].iterrows()],
            rotation=90
        )
        plt.ylabel('Score')
        plt.xlabel('Parameter Combination')
        plt.title('Top Parameter Combinations')
        plt.tight_layout()
        plt.show()
        
        # For each parameter, show distribution of scores
        param_names = [col for col in results_df.columns if col not in ['score', 'std']]
        
        fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 3*len(param_names)))
        
        for i, param in enumerate(param_names):
            ax = axes[i] if len(param_names) > 1 else axes
            
            # Group by parameter value and get mean score
            grouped = results_df.groupby(param)['score'].mean().reset_index()
            
            ax.bar(grouped[param].astype(str), grouped['score'])
            ax.set_title(f'Effect of {param} on Score')
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Score')
            
        plt.tight_layout()
        plt.show()

class CausalFilter:
    """
    A class for filtering causal inference results using TRS (Total Regulation Score) data.
    """

    def __init__(self, causal_inference: CausalInference, thres_S: float = 0.0, 
                 thres_R: float = 0.05, thres_TRS: float = 0.5, 
                 p_delta: float = 0.05, p_surrogate: float = 0.05):
        """
        Initialize the CausalFilter with parameters and a CausalInference instance.

        Args:
            causal_inference: An instance of the CausalInference class.
            thres_S: Threshold for S values.
            thres_R: Threshold for R values.
            thres_TRS: Threshold for TRS values.
            p_delta: P-value threshold for delta test.
            p_surrogate: P-value threshold for surrogate test.
        """
        self._ci = causal_inference
        self._thres_S = thres_S
        self._thres_R = thres_R
        self._thres_TRS = thres_TRS
        self._p_delta = p_delta
        self._p_surrogate = p_surrogate

        # Initialize data structures
        self._TRS_total = None
        self._delta_results = None
        self._surrogate_results = None

    @property
    def results(self):
        return {
            'TRS_total': self._TRS_total,
            'delta_results': self._delta_results,
            'surrogate_results': self._surrogate_results
        }

    def compute_TRS(self) -> np.ndarray:
        """
        Compute Target Regulation Score (TRS) for all component pairs and types.

        Returns:
            TRS_total: Array of TRS values with shape (num_pair, num_type).
        """
        start_time = time.time()

        # Access data from the CausalInference instance
        S_total_list = self._ci._results.get('S_total_list')
        R_total_list = self._ci._results.get('R_total_list')
        if S_total_list is None or R_total_list is None:
            raise ValueError("CausalInference results are incomplete. Please run the analysis first.")

        num_pair, num_type, num_data = S_total_list.shape

        # Initialize TRS matrix
        TRS_total = np.zeros((num_pair, num_type))

        # Compute TRS for each pair and type
        for i in range(num_pair):
            for j in range(num_type):
                S_tmp = S_total_list[i, j, :]
                R_tmp = R_total_list[i, j, :]

                # Apply thresholds
                S_processed = self.S_threshold(S_tmp, self._thres_S)
                R_processed = self.R_threshold(R_tmp, self._thres_R)

                # Calculate TRS
                if np.sum(R_processed) == 0:
                    TRS_total[i, j] = 0
                else:
                    TRS_total[i, j] = np.sum(S_processed * R_processed) / np.sum(R_processed)

        self._TRS_total = TRS_total
        print(f"TRS computation completed in {time.time() - start_time:.2f} seconds")

        return TRS_total

    def S_threshold(self, S_values: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply threshold to S values.

        Args:
            S_values: Array of S values.
            threshold: Threshold value.

        Returns:
            Processed S values.
        """
        result = np.copy(S_values)
        result[np.abs(result) < threshold] = 0
        return result

    def R_threshold(self, R_values: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply threshold to R values.

        Args:
            R_values: Array of R values.
            threshold: Threshold value.

        Returns:
            Boolean array where R values are above threshold.
        """
        return (R_values >= threshold).astype(float)

    def run_delta_test(self) -> Dict:
        """
        Run delta test on candidate pairs.

        Returns:
            Dictionary containing delta test results.
        """
        if self._TRS_total is None:
            self.compute_TRS()

        # Find candidates for delta test
        candidate_indices = np.where(self._TRS_total > self._thres_TRS)
        pair_indices = candidate_indices[0]
        type_indices = candidate_indices[1]

        delta_list = []
        for pair_idx, type_idx in zip(pair_indices, type_indices):
            # Extract S and R values for the candidate
            S_values = self._ci._results['S_total_list'][pair_idx, type_idx, :]
            R_values = self._ci._results['R_total_list'][pair_idx, type_idx, :]

            # Perform delta test (e.g., Wilcoxon signed-rank test)
            delta = S_values * R_values
            p_value = stats.wilcoxon(delta, alternative='greater').pvalue if len(delta) > 0 else 1.0

            delta_list.append((pair_idx, type_idx, p_value))

        # Filter results based on p_delta threshold
        delta_results = [d for d in delta_list if d[2] <= self._p_delta]
        self._delta_results = delta_results

        print(f"Delta test completed. Found {len(delta_results)} significant candidates.")
        return {'delta_results': delta_results}

    def run_surrogate_test(self, num_boot: int = 100) -> Dict:
        """
        Run surrogate test on candidates that passed the delta test.

        Args:
            num_boot: Number of bootstrap iterations.

        Returns:
            Dictionary containing surrogate test results.
        """
        if self._delta_results is None:
            self.run_delta_test()

        surrogate_results = []
        for pair_idx, type_idx, _ in self._delta_results:
            # Extract S and R values for the candidate
            S_values = self._ci._results['S_total_list'][pair_idx, type_idx, :]
            R_values = self._ci._results['R_total_list'][pair_idx, type_idx, :]

            # Perform surrogate test (e.g., bootstrap)
            surrogate_p_values = []
            for _ in range(num_boot):
                shuffled_S = np.random.permutation(S_values)
                surrogate_delta = shuffled_S * R_values
                surrogate_p = stats.wilcoxon(surrogate_delta, alternative='greater').pvalue if len(surrogate_delta) > 0 else 1.0
                surrogate_p_values.append(surrogate_p)

            # Aggregate surrogate p-values
            mean_p_value = np.mean(surrogate_p_values)
            surrogate_results.append((pair_idx, type_idx, mean_p_value))

        # Filter results based on p_surrogate threshold
        significant_results = [s for s in surrogate_results if s[2] <= self._p_surrogate]
        self._surrogate_results = significant_results

        print(f"Surrogate test completed. Found {len(significant_results)} significant candidates.")
        return {'surrogate_results': significant_results}

    def run_full_pipeline(self) -> Dict:
        """
        Run the complete pipeline: TRS computation, delta test, and surrogate test.

        Returns:
            Dictionary containing all results.
        """
        print("Starting full causal inference filtering pipeline...")

        # Step 1: Compute TRS
        self.compute_TRS()

        # Step 2: Run delta test
        self.run_delta_test()

        # Step 3: Run surrogate test
        self.run_surrogate_test()

        print("Pipeline completed!")
        return {
            'TRS_results': self._TRS_total,
            'delta_results': self._delta_results,
            'surrogate_results': self._surrogate_results
        }

def main():
    ci = CausalInference()
    
    # Display welcome message
    print("=========================================")
    print("  Causal Inference Analysis Tool")
    print("=========================================")
    
    # Load data
    file_path = input("Enter data file path (.mat, .csv, .xlsx, or .xls): ")
    ci.load_data(file_path)
    
    # Set dimension
    print(f"\nCurrent auto-detected dimension: {ci.dimension}")
    dimension_input = input("Enter dimension (1-3) or press Enter to use auto-detected: ")
    if dimension_input and dimension_input.isdigit():
        ci.dimension = int(dimension_input)
        print(f"Dimension set to: {ci.dimension}")
    
    # Parameter setup options
    print("\nParameter setup options:")
    print("1. Use default parameters")
    print("2. Enter parameters manually")
    print("3. Run hyperparameter tuning")
    param_choice = input("Choose an option (1-3): ")
    
    if param_choice == "2":
        ci.tune_parameters(method='interactive')
    elif param_choice == "3":
        ci.tune_parameters(method='auto')
    else:
        print("Using default parameters.")
    
    # Preprocess data
    print("\nPreprocessing data...")
    ci.preprocess()
    
    # Run analysis
    print("\nRunning causal inference analysis...")
    ci.run_analysis()
    
    # Initialize CausalFilter with the CausalInference instance
    cf = CausalFilter(
        causal_inference=ci,
        thres_S=0.1,
        thres_R=0.05,
        thres_TRS=0.5,
        p_delta=0.05,
        p_surrogate=0.05
    )
    
    # Run the full filtering pipeline
    print("\nRunning causal filtering pipeline...")
    filter_results = cf.run_full_pipeline()
    
    # Visualization options
    print("\nVisualization options:")
    print("1. TRS heatmap")
    print("2. Time Series Plot [placeholder function code, not yet fixed]")
    print("3. RDS scores across time windows")
    print("4. Component comparison")
    print("5. Time window segmentation")
    print("6. S-score heatmap")
    print("7. All visualizations")
    viz_choice = input("Choose an option (1-7): ")
    
    # Check if viz_choice is valid before converting to int
    if viz_choice.isdigit() and 1 <= int(viz_choice) <= 7:
        viz_choice = int(viz_choice)

        if viz_choice == 1 or viz_choice == 7:
            ci._visualizer.plot_trs_heatmap()
        if viz_choice == 2 or viz_choice == 7:
            ci._visualizer.plot_time_series()
        if viz_choice == 3 or viz_choice == 7:
            # Ask for specific components to visualize
            comp_input = input("Enter component indices to visualize (comma-separated) or press 0 for all: ")
            if comp_input:
                comp_indices = [int(idx) for idx in comp_input.split(',')]
                ci._visualizer.plot_rds_scores_time_windows(component_indices=comp_indices)
            else:
                ci._visualizer.plot_rds_scores_time_windows()
        if viz_choice == 4 or viz_choice == 7:
            ci._visualizer.plot_rds_component_comparison()
        if viz_choice == 5 or viz_choice == 7:
            ci._visualizer.plot_time_window_segmentation()
        if viz_choice == 6 or viz_choice == 7:
            ci._visualizer.plot_s_score_heatmap()
    else:
        print("Invalid visualization choice. Please enter a number between 1 and 7.")
    
    # Save results
    save_choice = input("\nSave results? (y/n): ")
    if save_choice.lower() == 'y':
        filename = input("Enter filename to save results (default: causal_inference_results.npy): ")
        if not filename:
            filename = f"causal_inference_results_dim{ci.dimension}.npy"
        ci.save_results(filename)
    
    # Save filtered results
    save_filter_choice = input("\nSave filtered results? (y/n): ")
    if save_filter_choice.lower() == 'y':
        filter_filename = input("Enter filename to save filtered results (default: causal_filtered_results.npy): ")
        if not filter_filename:
            filter_filename = "causal_filtered_results.npy"
        cf.save_results(filter_filename)

if __name__ == "__main__":
    main()