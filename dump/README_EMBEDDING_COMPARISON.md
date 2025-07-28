# Embedding-Based Resume-JD Comparison System

This system performs semantic comparison between resumes and job descriptions using embeddings instead of JSON extractions. It leverages vector similarity search to find the most relevant matches and provides detailed analysis.

## Overview

The new embedding-based approach offers several advantages over JSON-based comparison:

1. **Semantic Understanding**: Uses sentence transformers to capture semantic meaning rather than just keyword matching
2. **Better Similarity Detection**: Identifies conceptually similar content even with different wording
3. **Scalable**: Efficient vector similarity search for large datasets
4. **Flexible**: Supports targeted searches and various comparison modes

## Files

- `compare_main.py` - Main comparison script with embedding-based analysis
- `embedding_utils.py` - Utility functions for managing and analyzing embeddings
- `README_EMBEDDING_COMPARISON.md` - This documentation file

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install langchain langchain-community langchain-huggingface chromadb scikit-learn numpy
```

Also make sure you have:
- ChromaDB stores populated with embeddings (`chroma_db_jd` and `chroma_db_resume`)
- Ollama running with the specified model (`bigllama/mistralv01-7b:latest`)

## Usage

### 1. Check System Status

Before running comparisons, check if your embedding stores are properly set up:

```bash
python embedding_utils.py --status
```

This will show:
- Whether embedding stores exist
- Number of documents in each store
- Available result files

### 2. Run Full Comparison

Run a comprehensive comparison between all resumes and job descriptions:

```bash
python compare_main.py --mode full
```

Options:
- `--threshold 0.7` - Set similarity threshold (default: 0.7)
- `--top-k 10` - Number of top matches to analyze in detail (default: 10)

### 3. Run Targeted Search

Search for specific skills, roles, or requirements:

```bash
# Search both resumes and JDs
python compare_main.py --mode search --query "machine learning engineer"

# Search only resumes
python compare_main.py --mode search --query "python tensorflow" --search-type resume

# Search only job descriptions
python compare_main.py --mode search --query "generative AI" --search-type jd
```

### 4. Analyze Embedding Quality

Check the quality and distribution of your embeddings:

```bash
python embedding_utils.py --analyze-quality
```

### 5. Search Similar Documents

Find documents similar to a specific query:

```bash
python embedding_utils.py --search "deep learning experience" --doc-type both
```

### 6. Analyze Previous Results

Analyze results from a previous comparison run:

```bash
python embedding_utils.py --analyze-results output/embedding_comparison_results.json
```

## Output

### Full Comparison Results

The full comparison generates a JSON file with:

```json
{
  "comparison_type": "embedding_based",
  "timestamp": "2024-01-15 10:30:00",
  "processing_time_seconds": 45.2,
  "total_resumes": 7,
  "total_jds": 1,
  "total_matches_above_threshold": 5,
  "similarity_threshold": 0.7,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "top_matches": [
    {
      "rank": 1,
      "resume_id": "candidate_name.json",
      "jd_id": "job_posting.json",
      "embedding_similarity": 0.8542,
      "detailed_analysis": {
        "skills_match": {
          "score": 92.5,
          "analysis": "Strong alignment in ML/AI skills..."
        },
        "experience_match": {
          "score": 88.0,
          "analysis": "Relevant experience in similar roles..."
        },
        "education_match": {
          "score": 95.0,
          "analysis": "Educational background exceeds requirements..."
        },
        "role_compatibility": {
          "score": 90.0,
          "analysis": "Role alignment is excellent..."
        },
        "overall_match": {
          "score": 91.4,
          "analysis": "Excellent candidate with strong technical fit..."
        },
        "recommendation": "hire - strong technical and cultural fit"
      }
    }
  ],
  "all_similarity_scores": [...]
}
```

### Search Results

Targeted searches return documents ranked by similarity:

```json
{
  "query": "machine learning engineer",
  "search_type": "both",
  "timestamp": "2024-01-15 10:35:00",
  "results": [
    {
      "type": "resume",
      "id": "candidate_name.json",
      "similarity_score": 0.8901,
      "content": "Skills: Python, TensorFlow, PyTorch...",
      "metadata": {"source": "candidate_name.json", "job_role": "ML Engineer"}
    }
  ]
}
```

## Key Differences from JSON-Based Comparison

| Aspect | JSON-Based | Embedding-Based |
|--------|------------|-----------------|
| **Comparison Method** | Direct JSON field comparison | Vector similarity search |
| **Semantic Understanding** | Limited to exact/similar keywords | Captures semantic meaning |
| **Flexibility** | Fixed field structure | Flexible content analysis |
| **Performance** | Fast for small datasets | Scalable for large datasets |
| **Accuracy** | Depends on extraction quality | Better semantic matching |

## Configuration

Key configuration options in `compare_main.py`:

```python
# Paths
JD_DB_PATH = "chroma_db_jd"
RESUME_DB_PATH = "chroma_db_resume"

# Models
MODEL_NAME = "bigllama/mistralv01-7b:latest"  # LLM for detailed analysis
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model

# Thresholds
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for matches
```

## Troubleshooting

### Common Issues

1. **No embeddings found**
   - Ensure ChromaDB stores exist and contain data
   - Run the embedding creation scripts in the `Embedding/` directory

2. **Low similarity scores**
   - Try lowering the similarity threshold
   - Check if embeddings were created correctly

3. **LLM connection errors**
   - Ensure Ollama is running
   - Verify the model name is correct

4. **Memory issues**
   - Reduce the number of documents processed at once
   - Use a smaller embedding model if needed

### Performance Tips

1. **For large datasets**: Increase similarity threshold to reduce processing time
2. **For better accuracy**: Use a more sophisticated embedding model
3. **For faster processing**: Reduce the number of detailed analyses (`--top-k`)

## Integration with Existing System

This embedding-based system can work alongside your existing JSON-based comparison:

1. **Parallel Operation**: Run both systems and compare results
2. **Hybrid Approach**: Use embeddings for initial filtering, JSON for detailed analysis
3. **Migration**: Gradually replace JSON-based comparisons with embedding-based ones

## Future Enhancements

Potential improvements:
- Support for multiple embedding models
- Real-time comparison API
- Advanced filtering and ranking options
- Integration with external job boards
- Custom embedding fine-tuning for domain-specific matching
