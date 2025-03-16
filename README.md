# Transformer-based Pitch Recommendation System

## Project Overview

This project develops an intelligent baseball pitch recommendation system using advanced deep learning techniques. By combining transformer architectures with reinforcement learning, I aim to model complex pitch sequencing strategies in baseball and provide optimal pitch recommendations based on game context.

## Current Status

- **Phase 1 ✓**: Data Collection & Processing Pipeline
- **Phase 2 ✓**: Feature Engineering & Embedding Generation
- **Phase 3 ✓**: Transformer Pretraining (Next-Pitch Prediction)
  - Achieved ~70% top-1 prediction accuracy
- **Phase 4 ⟳**: Reinforcement Learning Integration (In Progress)
- **Phase 5**: Scaling & Optimization (Planned)

## Technical Architecture


The system consists of four main components:

1. **Data Processing Pipeline**: Collects and processes MLB Statcast data, engineered with baseball-specific features.

2. **Pitch Embedding Model**: Converts raw pitch data to dense vector representations that capture pitch characteristics, game context, and situational features.

3. **Transformer Model**: Pretrained on next-pitch prediction to understand sequential patterns in pitch selection.

4. **Reinforcement Learning Module**: Will use the transformer's representations to optimize pitch selection based on defined reward signals.

## Results

Current prototype shows promising results with the pretraining task:
- ~70% top-1 accuracy in next pitch prediction
- Model successfully captures pitcher tendencies and situational patterns

## Repository Structure

- **Configs**: Parameter configurations and file paths
- **Data**: 
  - `/raw`: Raw Statcast data (2021-2024)
  - `/processed`: Cleaned and feature-engineered datasets
  - `/embeddings`: Generated pitch embeddings
- **Notebooks**:
  - `data-collection.ipynb`: Statcast data acquisition
  - `cleaning.ipynb`: Data cleaning processes
  - `engineering.ipynb`: Feature engineering
  - `embedding.ipynb`: Generating contextual pitch embeddings
  - `pretraining.ipynb`: Training transformer on next-pitch prediction
- **Src**: Python modules
  - `data_collection.py`: MLB Statcast API functions
  - `data_cleaning.py`: Data preprocessing functions
  - `feature_engineering.py`: Baseball-specific feature extraction
  - `embeddings.py`: Pitch embedding models
  - `transformer.py`: Transformer architecture definitions
  - `pretraining.py`: Transformer training utilities
- **Models**: Saved model checkpoints
- **Figures**: Training visualizations and results plots
- **Literature**: Relevant papers on transformers and RL

## Next Steps

1. **Develop RL Environment**: Create a baseball environment that simulates pitch outcomes
2. **Implement RL Agent**: Design a policy gradient agent that leverages transformer representations
3. **Design Reward Functions**: Create baseball-specific reward signals capturing both immediate outcomes and strategic considerations
4. **Prototype Integration**: Test integration of transformer and RL components on sample data
5. **Scale Training**: Move to more powerful hardware and train on full dataset
6. **Evaluation**: Benchmark against traditional and heuristic methods

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/username/pitch-recommendation.git
cd pitch-recommendation

# Create virtual environment
conda create -n pitch-rec python=3.9
conda activate pitch-rec

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Collection
```python
from src.data_collection import pitcher_year_month_statcast

# Collect data for a specific month
pitcher_year_month_statcast(2023, 6)
```

### Training Pipeline
```python
# See notebooks/pretraining.ipynb for full training workflow
```

## Dependencies

- PyTorch 2.0+
- pandas
- numpy
- pybaseball
- matplotlib
- scikit-learn
- tqdm

## References

- Vaswani, A., et al. (2017). Attention is All You Need.
- [MLB Statcast Documentation](https://baseballsavant.mlb.com/statcast_search)