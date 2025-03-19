# General ODE-based Inference

The **General ODE-based Inference (GOBI)** is a Python-based framework for analyzing causal relationships in time-series data. It provides functionality for preprocessing data, running causal inference analysis, filtering results, and visualizing the outcomes. The tool is designed to handle multi-dimensional causal analysis and supports hyperparameter tuning for optimal performance.

This is a Python translation of the MATLAB package found in https://github.com/Mathbiomed/GOBI. The methodology and approach are inspired by the work described in [Nature Communications](https://www.nature.com/articles/s41467-023-39983-4).

---

## Features

- **GOBI Analysis**: Analyze causal relationships using Regulation Detection Scores (RDS) and Total Regulation Scores (TRS).
- **Filtering Pipeline**: Includes delta and surrogate tests to filter significant causal relationships.
- **Visualization**: Generate heatmaps, time-series plots, and other visualizations to interpret results.
- **Hyperparameter Tuning**: Supports both interactive and automated tuning of analysis parameters.
- **Data Preprocessing**: Handles time-series data with moving averages, interpolation, and normalization.
- **Based on Published Research**: The methodology is inspired by the work described in [Nature Communications](https://www.nature.com/articles/s41467-023-39983-4).

---

## Installation

1. Clone the repository or download the `GOBI_analysis_pipeline.py` file.
2. Install the required Python libraries:
   ```bash
   pip install numpy matplotlib scipy pandas numba tqdm joblib
   ```

---

## Usage

### Running the Tool

1. Run the script:
   ```bash
   python GOBI_analysis_pipeline.py
   ```

2. Follow the prompts to:
   - Load your data file (`.mat`, `.csv`, `.xlsx`, or `.xls`). Copy the file's path.
   - Set the analysis dimension (1-3).
   - Configure parameters (default, manual, or hyperparameter tuning).
   - Preprocess the data.
   - Run the causal inference analysis.
   - Filter the results using delta and surrogate tests.
   - Visualize the results.

3. Save the results to `.npy` files for future use.

---

## Input Data Format

The tool supports the following file formats:
- **MATLAB (`.mat`)**: Requires `t` (time) and `y` (data) arrays.
- **CSV (`.csv`)**: The first column should contain time points, and subsequent columns should contain data.
- **Excel (`.xlsx` or `.xls`)**: The first column should contain time points, and subsequent columns should contain data.

---

## Key Classes and Functions

### `CausalInference`
- **`load_data(file_path)`**: Load time-series data from a file.
- **`preprocess()`**: Preprocess the data (moving average, interpolation, normalization).
- **`run_analysis()`**: Perform causal inference analysis.
- **`tune_parameters(method='interactive')`**: Tune analysis parameters interactively or automatically.
- **`plot_trs_heatmap()`**: Visualize the TRS heatmap.
- **`plot_time_series()`**: Plot the time-series data.

### `CausalFilter`
- **`compute_TRS()`**: Compute Total Regulation Scores (TRS).
- **`run_delta_test()`**: Perform delta tests to filter significant causal relationships.
- **`run_surrogate_test(num_boot=100)`**: Perform surrogate tests for further filtering.
- **`run_full_pipeline()`**: Run the complete filtering pipeline (TRS, delta test, surrogate test).

---

## Visualization Options

The tool provides several visualization options:
1. **TRS Heatmap**: Visualize the Total Regulation Scores.
2. **Time-Series Plot**: Plot the preprocessed time-series data.
3. **RDS Scores Across Time Windows**: View Regulation Detection Scores over time.
4. **Component Comparison**: Compare S-scores for different regulation types.
5. **Time Window Segmentation**: Visualize segmented time windows.
6. **S-Score Heatmap**: Heatmap of S-scores across time windows.

---

## Example Workflow

1. **Load Data**:
   Provide the path to your data file (e.g., `data.csv`).

2. **Set Parameters**:
   Choose default parameters, manually set them, or run hyperparameter tuning.

3. **Run Analysis**:
   Perform causal inference analysis on the preprocessed data.

4. **Filter Results**:
   Use delta and surrogate tests to identify significant causal relationships.

5. **Visualize Results**:
   Select from various visualization options to interpret the results.

6. **Save Results**:
   Save the analysis and filtered results for future use.

---

## Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `pandas`
  - `numba`
  - `tqdm`
  - `joblib`

---

## File Structure

- **`GOBI_analysis_pipeline.py`**: Main script containing the `CausalInference` and `CausalFilter` classes.
- **Input Data**: Time-series data in `.mat`, `.csv`, `.xlsx`, or `.xls` format.
- **Output Files**: Results saved as `.npy` files.

---

## Future Improvements

- Add support for additional data formats.
- Enhance visualization options with interactive plots.
- Optimize performance for large datasets.
- **Test for edge cases**.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or feedback, please contact ash.lastimosa@gmail.com.

