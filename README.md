# Research Paper Classification System

A comprehensive machine learning pipeline for classifying research papers into scientific disciplines using a fine-tuned BERT model. This system includes web scraping capabilities, data preprocessing, and a high-accuracy classification model.

## Quick Start

### Prerequisites
- Google Colab (recommended) or Jupyter Notebook
- Google Drive account for data storage

### Installation & Setup

1. **Clone this repository** to your Colab environment:
```python
!git clone https://github.com/Emran025/Research_Paper_Classification_model.git
%cd Research_Paper_Classification_model
```

2. **Download all project assets** using our downloader notebook:

Open and run `gdrive_folder_downloader.ipynb` to download:
- Pre-trained BERT model (`bert_text_classifier`)
- Training checkpoints and logs
- Cleaned dataset (`cleaned_dataset.csv`)
- Raw scraped research papers

3. **You're ready to go!** All project components are now available locally.

## 📁 Project Structure

```
research-paper-classification/
├── gdrive_folder_downloader.ipynb      # Download project assets from Google Drive
├── mendeley_scraper.ipynb              # Scrape research papers from Mendeley
├── research_papers_bert_finetuning.ipynb # BERT model training pipeline
├── bert_text_classifier/               # Pre-trained model (downloaded)
├── bert_classification_output/         # Training artifacts (downloaded)
├── cleaned_dataset.csv                 # Processed dataset (downloaded)
└── Mendeley_Research/                  # Raw scraped data (downloaded)
```

## Usage Options

### Option 1: Use Pre-trained Model (Recommended)
After running the downloader notebook, you'll have immediate access to:
- Fine-tuned BERT model (95.39% accuracy)
- Pre-processed dataset
- Training logs and checkpoints

### Option 2: Full Pipeline from Scratch
For complete reproducibility:

1. **Data Collection**: Run `mendeley_scraper.ipynb` to scrape fresh data from Mendeley
2. **Model Training**: Execute `research_papers_bert_finetuning.ipynb` to fine-tune BERT
3. **Evaluation**: Use the trained model for inference and analysis

## Model Performance

The fine-tuned BERT model achieves state-of-the-art performance:

- **Evaluation Loss**: 0.184
- **Overall Accuracy**: 95.39%

### Detailed Classification Report

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Biology | 0.94 | 0.93 | 0.94 | 3,177 |
| Business | 0.96 | 0.97 | 0.97 | 3,179 |
| Chemistry | 0.94 | 0.96 | 0.95 | 3,073 |
| Computer Science | 0.96 | 0.93 | 0.95 | 2,987 |
| Environmental Science | 0.95 | 0.94 | 0.95 | 2,850 |
| Mathematics | 0.93 | 0.96 | 0.95 | 3,091 |
| Medicine | 0.97 | 0.96 | 0.96 | 3,067 |
| Physics | 0.97 | 0.95 | 0.96 | 3,181 |
| Psychology | 0.97 | 0.97 | 0.97 | 3,348 |

## Hugging Face Integration

The model is also available on Hugging Face Hub for easy integration:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Emran025/bert_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("Emran025/bert_text_classifier")
```

**Model Card**: [https://huggingface.co/Emran025/bert_text_classifier](https://huggingface.co/Emran025/bert_text_classifier)

## Data Sources

All datasets are organized by discipline and available via Google Drive:
- Computer Science, Medicine, Business, Chemistry, Mathematics
- Psychology, Environmental Science, Biology, Physics

##  Technical Specifications

- **Framework**: PyTorch with Transformers
- **Base Model**: BERT (bert-base-uncased)
- **Training Environment**: Google Colab
- **Dataset Size**: ~27,000 research papers
- **Classes**: 9 scientific disciplines

##  Note

This project is optimized for Google Colab. All file paths and dependencies are configured for Colab's environment. When running locally, adjust the file paths accordingly.

---

**Contributors**: Emran025  
**License**: MIT  
**Last Updated**: OCT 2025

