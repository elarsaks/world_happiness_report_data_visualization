# Data Pipeline Diagram

<br></br>
<br></br>

```mermaid
flowchart LR
    A[("Raw Data<br/>(UNSDSN)")]
    B["Preprocessing Notebook"]
    C[("Processed Dataset<br/>(CSV)")]
    D["Analytics Notebook"]
    E["Tableau Project"]

    A -->|"Read raw data"| B
    B -->|"Normalize & Merge"| C
    B -->|"Development input"| D
    C -->|"Standalone dataset"| D
    C -->|"Standalone dataset"| E

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#fce4ec
    style E fill:#f3e5f5
```

<br></br>
<br></br>
## Pipeline Steps

1. **Raw Data Ingestion**: Read raw data published by (NAME)
2. **Data Preprocessing**: Normalize and merge data in Kaggle notebook
3. **Processed Dataset**: Output single CSV file, published as standalone Kaggle dataset
4. **Downstream Consumption**:
   - Analytics notebook (development used upstream notebook directly)
   - Tableau project (uses published standalone dataset)

## Links

| Component | Link |
|-----------|------|
| Raw Data | (LINK) |
| Preprocessing Notebook | (LINK) |
| Processed Dataset | (LINK) |
