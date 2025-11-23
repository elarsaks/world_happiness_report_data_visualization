# World Happiness Report Data Visualization

Explore how happiness scores evolved worldwide between 2015 and 2019. The project ships with raw data, a ready-to-use processed dataset, and a notebook that walks through the analysis and visualizations.

## What's in the Folder

```
world_happiness_report_data_visualization/
├── raw_data/                    # Original CSV files for each year
├── happiness_combined_2015_2019.csv  # Cleaned dataset produced by the script/notebook
├── data_preprocessor.py         # Script that regenerates the processed CSV when needed
├── Notebook.ipynb               # Main analysis notebook
└── README.md                    # You're reading it
```

## Getting the Project (No Git Needed)

1. On GitHub, click the green **Code** button and choose **Download ZIP**.
2. Unzip the download somewhere convenient (e.g., Desktop or Documents).
3. Open the unzipped folder in Finder/Explorer or directly in VS Code.

If you already use Git, `git clone` works too—see the note at the end of this document.

## Requirements

- Python 3.8 or newer (Anaconda, Miniconda, or python.org installers all work).
- The following Python packages: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn.

To install the packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

If `pip` is not recognised, open the Anaconda Prompt (Windows) or Terminal (macOS/Linux) and run the same command there.

## Running the Notebook

1. Open a terminal in the project folder. In VS Code you can use **Terminal → New Terminal**.
2. (Optional but recommended) Create and activate a virtual environment so packages stay isolated.
3. Launch Jupyter:

	```bash
	jupyter notebook Notebook.ipynb
	```

4. When the notebook opens in your browser, run the first code cell. It checks for `happiness_combined_2015_2019.csv` and regenerates it if missing.
5. Continue running the remaining cells in order (use the **Run** ▶ button or press **Shift+Enter**). The notebook will produce all charts and tables used in the analysis.

## Regenerating the Processed CSV Without the Notebook

If you prefer a one-line command, run the following from the project folder:

```bash
python -c "from data_preprocessor import process_and_save_data; process_and_save_data()"
```

This rebuilds `happiness_combined_2015_2019.csv` using the embedded metadata and raw CSV files.

## Troubleshooting

- **Plotly import errors**: make sure `plotly` is installed (`pip install plotly`).
- **Permission denied**: if the processed CSV is open in Excel or another program, close it and rerun the cell or command.
- **Python not found**: confirm Python is on your PATH or launch the notebook through Anaconda Navigator.

## Optional: Using Git

For teammates comfortable with Git:

```bash
git clone https://github.com/elarsaks/world_happiness_report_data_visualization.git
cd world_happiness_report_data_visualization
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

## Data Source

World Happiness Report data (2015–2019) downloaded from the [Kaggle dataset](https://www.kaggle.com/datasets/unsdsn/world-happiness?resource=download).


