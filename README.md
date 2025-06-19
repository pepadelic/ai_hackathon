# 🎬 Movie Plot Generator

A Streamlit application for generating original movie plots based on selected genre and positive aspects from movie reviews.

## ✨ Features

- 📄 **PDF Upload & Processing**: Extract text from PDF with movie reviews
- 🎭 **Genre Selection**: 15 different movie genres to choose from
- 🔍 **Intelligent Search**: Semantic search for positive aspects from reviews
- 📝 **Plot Generation**: AI creates original plots in Czech
- 🎬 **Review Inspiration**: Plots inspired by what people appreciate in movies
- 📚 **Plot History**: Storage of all generated plots

## 🚀 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run movie_plot_generator.py
   ```
3. **Open the application in web browser (the URL provided by streamlit run command)**

## 📖 How to Use

### 1. Document Upload
- Click on "Choose PDF file" in the sidebar
- Upload PDF with movie reviews
- Wait for document processing

### 2. Genre Selection
After document processing:
- Select one of 15 available genres using radio buttons:
  - Action, Comedy, Drama, Horror, Sci-Fi, Fantasy
  - Romance, Thriller, Crime, Historical, War
  - Biography, Adventure, Western, Mystery

### 3. Plot Generation
- Click on "🎬 Generate Plot"
- Application finds relevant positive aspects from reviews
- AI generates original plot in Czech

### 4. Plot Management
- All generated plots are displayed with timestamps
- You can clear all plots with "🗑️ Clear All Plots" button

## 🎯 How It Works

### Architecture
```
PDF Reviews → Text Extraction → Text Chunking → Embeddings → Vector Database
                                                                    ↓
Genre Selection → Semantic Search → Review Context → AI Generation → Original Plot
```

### Components

- **PDF Processing**: PyMuPDF, pdfplumber and PyPDF2 for robust text extraction
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2) for semantic representations
- **Vector Database**: FAISS for fast similarity search
- **AI Integration**: Granite-32-8B-Instruct model for plot generation

### Configuration

The application is configured for:
- **LLM Endpoint**: Granite-32-8B-Instruct
- **Embeddings**: all-MiniLM-L6-v2
- **Chunk Size**: 1000 characters with 200 character overlap
- **AI Temperature**: 0.8 for creative generation
- **Max Tokens**: 1500 for longer plots

## 📋 System Prompt

AI uses the following instructions:
- Generate ONLY in Czech
- Create original plots (not copies of existing movies)
- Plots should be 200-300 words long
- Include main characters, conflict and hint of resolution
- Be inspired by positive aspects from reviews
- Do not mention specific names of existing movies

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies from requirements.txt are installed
2. **PDF Errors**: Application uses 3 different libraries for PDF extraction as backup
3. **API Connection**: Check LLM endpoint availability
4. **Memory Issues**: For large documents consider smaller PDF files

### Performance Tips

- **Smaller PDFs**: Work better for faster processing
- **Clear Text**: PDFs with readable text work best
- **Internet Connection**: Required for model downloads and API calls

## 📚 Dependencies

- streamlit: Web interface
- PyPDF2, pdfplumber, PyMuPDF: PDF text extraction
- sentence-transformers: Text embeddings
- faiss-cpu: Vector search
- numpy: Numerical operations
- requests: HTTP API calls
- torch: Deep learning framework
- scikit-learn: Fallback embeddings

## 🎬 Sample Genres

The application supports these movie genres:
- **Action**: Movies full of action and tension
- **Comedy**: Funny and entertaining movies
- **Drama**: Emotionally rich stories
- **Horror**: Scary and suspenseful movies
- **Sci-Fi**: Science fiction stories
- **Fantasy**: Fantastic worlds and magic
- **Romance**: Love stories
- **Thriller**: Suspenseful psychological movies
- **Crime**: Detective and criminal stories
- **Historical**: Movies from various historical periods
- **War**: Stories from wars and conflicts
- **Biography**: Stories of real people
- **Adventure**: Movies full of adventure
- **Western**: Stories from the Wild West
- **Mystery**: Mysterious and enigmatic stories

## 📄 License

This project is intended for educational and demonstration purposes. 
