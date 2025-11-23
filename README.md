# World Happiness Report Data Visualization

This project explores how global happiness has evolved across five years of World Happiness Report data (2015-2019). The analysis focuses on comparing countries and regions, uncovering drivers of happiness, and highlighting notable shifts in well-being around the world.

## Project Structure

```
world_happiness_report_data_visualization/
├── raw_data/                    # Original CSV files for each year
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   └── 2019.csv
├── metadata/                    # Data normalization mappings
│   ├── column_normalization.csv
│   ├── country_aliases.csv
│   ├── region_overrides.csv
│   ├── continent_overrides.csv
│   └── region_to_continent.csv
├── processed_data/             # Cleaned and combined dataset
│   └── happiness_combined_2015_2019.csv
├── src/                        # Data processing modules
│   ├── __init__.py
│   ├── data_loader.py         # Functions for loading processed data
│   ├── data_normalizer.py     # Data cleaning and normalization
│   └── utils.py               # Utility functions
├── Notebook.ipynb             # Main analysis notebook
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

## Usage

### Data Processing

The raw data has already been processed and saved to `processed_data/happiness_combined_2015_2019.csv`. If you need to regenerate the processed data:

```python
from src.data_loader import process_and_save_data
process_and_save_data()
```

### Running the Analysis

1. Open the notebook:
```bash
jupyter notebook Notebook.ipynb
```

2. The notebook will automatically load the processed data and guide you through the analysis.

## Research Questions

- How do global happiness scores change from 2015 to 2019?
- Which countries improved or declined the most across the five-year span?
- How do regions and continents differ in their happiness trajectories?
- Which factors show the strongest relationships with happiness over time?

## Key Findings

- Global happiness levels remained remarkably stable between 2015 and 2019
- Western Europe and North America consistently lead the rankings
- Eastern European countries showed the most improvement
- GDP per capita, social support, and healthy life expectancy are the strongest predictors of happiness

## Data Sources

World Happiness Report data from 2015-2019, available from the [World Happiness Report website](https://worldhappiness.report/).

## License

This project is for educational and research purposes.
