🚦 Traffic Congestion Prediction using GAT + LSTM
This project predicts urban traffic congestion using a hybrid deep learning model combining:
Graph Attention Networks (GAT) → spatial relationships between sensors
LSTM (Long Short-Term Memory) → temporal traffic patterns
It includes a Flask dashboard for visualization.
---
📊 Features
Multi-dataset support:
PEMS-BAY
Caltrans D7
Seattle Loop
End-to-end pipeline:
Data loading
Preprocessing
Model training
Prediction
Visualization:
Actual vs Predicted plots
Traffic congestion heatmap (Leaflet)
---
🧠 Model Architecture
GAT + LSTM Pipeline
Input (Traffic Speeds)
↓
GAT Layer (Spatial Learning)
↓
Flatten
↓
LSTM (Temporal Learning)
↓
Dense Layer
↓
Predicted Traffic Speed

---
📁 Project Structure
MLProject/
│
├── app.py # Flask app
├── pipeline.py # Core pipeline logic
├── model.py # GAT + LSTM model
├── utils.py # Data utilities
│
├── data/
│ ├── bay/
│ ├── d7/
│ └── seattle/
│
├── processed/ # Generated data
├── static/ # Plots + heatmap JSON
├── templates/
│ └── index.html
│
└── requirements.txt
---
⚙️ Installation
1. Clone repository
```bash
git clone <your-repo-url>
cd MLProject

2. Create virtual environment

python -m venv venv
venv\Scripts\activate    # Windows

3. Install dependencies

pip install -r requirements.txt

▶️ Running the Application

python app.py

Open browser:

http://127.0.0.1:5000

🔄 Pipeline Steps

When you click Run:

a) Data Conversion
b) Subset Extraction
c) Model Training
d) Prediction & Evaluation
e) Heatmap Generation

📈 Outputs
1. Prediction Plot
Shows actual vs predicted traffic speed
Helps evaluate temporal accuracy

2. Heatmap
Displays congestion across locations
Color scale:
🟢 Green → Free flow
🟡 Yellow → Moderate congestion
🟠 Orange → Heavy congestion
🔴 Red → Severe congestion

3. Metrics
Metric	Meaning
MAE	Average absolute error
RMSE	Penalizes large errors

📌 Reproducibility

Random seeds are fixed in training:

import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

⚠️ Known Limitations
No adjacency matrix → GAT learns fully-connected graph
No external features (weather, events)
Limited training epochs (default: 5)
Heatmap depends heavily on dataset quality

🚀 Future Improvements
Add real road graph (adjacency matrix)
Use Graph Convolution Networks (GCN)
Add weather/event features
Deploy on cloud (Docker + Gunicorn)
Add live traffic API

👨‍💻 Author

Developed as part of ML Project using GAT + LSTM architecture.

📜 License

For academic use only.