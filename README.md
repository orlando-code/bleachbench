# [ICCS 2025 Hackathon](https://github.com/Cambridge-ICCS)


## 🎯 Project Overview
**Exploring the comparative predictive power of traditional and ML methods for coral bleaching**


## 📁 Project Structure

```
coral_bleaching_prediction/
├── data/                   # Data files (raw and processed)
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
│   ├── dev/               # Rapid prototyping notebooks
│   ├── presentation       # Communicating methods and results
├── tests/                 # Unit tests
├── config/                # Configuration files
├── results/               # Model outputs and visualizations
└── docs/                  # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/orlando-code/bleachbench
cd bleachbench
```

2. Create a virtual environment using venv (just to keep things consistent between users: other package managers could be added later):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data

The project uses time series data related to coral bleaching events. Data should be placed in the `data/` directory. Data files will be made available on Zenodo.

## 🔬 Models

### Traditional Methods
- Degree heating weeks (DHWs)

### Proposed machine Learning Methods
- Random Forest
- XGBoost
- LSTM (Long Short-Term Memory)

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Directional Accuracy

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request to merge your changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- Orlando Timmerman (✉️ rt582@cam.ac.uk)
- Akash Vermo
- Robert Rouse
- Matt Archer

## 📚 References

TO DO

## 🤝 Acknowledgments

Many thanks to the ICCS team who have made today – and the amazing week of activities – possible!