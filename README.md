# Adobe India Hackathon 2025: Connecting the Dots - PDF Document Intelligence

## Project Overview

This project is submitted for the Adobe India Hackathon 2025, addressing the "Connecting the Dots" challenge. [cite_start]The primary goal of Round 1A is to transform static PDFs into intelligent, interactive experiences by extracting structured outlines (Title, H1, H2, H3 headings) in a clean, hierarchical JSON format[cite: 493, 506]. This forms the foundation for richer document understanding and interaction. [cite_start]For Round 1B, the system aims to act as an intelligent document analyst, extracting and prioritizing relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## Round 1A: Understand Your Document

### Mission
[cite_start]To extract a structured outline (Title, H1, H2, H3 with level and page number) from a given PDF file and output it as a valid JSON file[cite: 493, 505, 506].

### Output Format Example
```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2},
    { "level": "H3", "text": "History of AI", "page":3}
  ]
}