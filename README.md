# Research Paper Classification System

A comprehensive machine learning pipeline for classifying research papers into scientific disciplines using a fine-tuned BERT model. This system includes web scraping capabilities, data preprocessing, and a high-accuracy classification model.

## Target Scientific Disciplines

The system classifies research papers into 9 major scientific categories with detailed subfields:

| Category | Subfields |
|----------|-----------|
| **Computer Science** | Artificial Intelligence, Cloud Computing, Cyber Security, Quantum Computing, Software Engineering |
| **Medicine** | Cardiology, Surgery, Neurology, Oncology, Pediatrics, Psychiatry |
| **Business** | Marketing, Finance, Management, Entrepreneurship, E-commerce |
| **Biology** | Genetics, Ecology, Zoology, Biochemistry, Physiology |
| **Physics** | Quantum Mechanics, Astrophysics, Particle Physics, Cosmology |
| **Chemistry** | Organic Chemistry, Analytical Chemistry, Biochemistry, Medicinal Chemistry |
| **Mathematics** | Algebra, Calculus, Statistics, Probability, Optimization |
| **Psychology** | Clinical Psychology, Cognitive Psychology, Social Psychology, Neuropsychology |
| **Environmental Science** | Climate Change, Conservation, Sustainability, Renewable Energy |

## Dataset Statistics

### Data Collection & Cleaning
- **Original Records**: 155,882 research papers
- **Clean Non-duplicate Records**: 140,004 papers
- **Cleaning Ratio**: 89.81%

### Final Dataset Distribution
- **Psychology**: 16,821 records (12.0%)
- **Chemistry**: 16,675 records (11.9%)
- **Physics**: 15,941 records (11.4%)
- **Business**: 15,929 records (11.4%)
- **Mathematics**: 15,464 records (11.0%)
- **Medicine**: 15,361 records (11.0%)
- **Computer Science**: 14,776 records (10.6%)
- **Biology**: 14,729 records (10.5%)
- **Environmental Science**: 14,308 records (10.2%)

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

## Project Structure

```
Research_Paper_Classification_model/
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
- Pre-processed dataset (140,004 research papers)
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
- **Evaluation Runtime**: 428.03 seconds
- **Samples Processed**: 65.306 samples/second

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

## Model Integration

### Hugging Face Hub
The model is available on Hugging Face Hub for easy integration:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Emran025/bert_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("Emran025/bert_text_classifier")
```

**Model Card**: [https://huggingface.co/Emran025/bert_text_classifier](https://huggingface.co/Emran025/bert_text_classifier)

### Google Drive Resources
All datasets and model artifacts are available via Google Drive:
- **Main Drive Folder**: [Google Drive Link](https://drive.google.com/drive/folders/1EksEOjw5IzGjJv1hFOh9lqvtJy7trk_W?usp=drive_link)
- Raw datasets organized by scientific discipline
- Pre-trained model and training checkpoints
- Cleaned and processed datasets

## Data Sources

Data was collected from **Mendeley** research catalog across 9 major disciplines:
- Computer Science, Medicine, Business, Chemistry, Mathematics
- Psychology, Environmental Science, Biology, Physics

Each category contains multiple specialized subfields for comprehensive coverage.

## Technical Specifications

- **Framework**: PyTorch with Transformers
- **Base Model**: BERT (bert-base-uncased)
- **Training Environment**: Google Colab
- **Dataset Size**: 140,004 research papers (after cleaning)
- **Classes**: 9 scientific disciplines with 5-12 subfields each
- **Training Samples**: 27,953 papers (evaluation set)

## Note

This project is optimized for Google Colab. All file paths and dependencies are configured for Colab's environment. When running locally, adjust the file paths accordingly.

---

**Contributors**: Emran Nasser, Mohammed Alyafrosy, Ryadh Alizi  
**License**: MIT  
**Last Updated**: October 2024
