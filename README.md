# Descriptive Writing Visualiser

An educational web tool designed to motivate primary school children 
to write more descriptively by rewarding richer writing with 
higher-quality AI-generated images.

## What it does

Children type a description into the app. The app scores their writing 
based on vocabulary richness, sentence variety, adjective use, and 
sensory detail. The score then determines the quality of the AI image 
generated from their text — better writing produces a more detailed, 
colourful image.

This creates a feedback loop that encourages children to keep improving 
their writing.

## Features

- NLP-based writing scorer using NLTK (POS tagging, tokenisation)
- Dynamic image prompt adjustment based on writing score
- Child safety filter (blocks violence, sexual content, personal info)
- Attempt tracking with score comparison across tries
- Anonymous logging of attempt metrics to CSV
- Built with Streamlit for a simple classroom-friendly interface

## Tech stack

- Python
- Streamlit
- OpenAI API (image generation)
- NLTK (natural language processing)

## How to run

1. Clone the repo
2. Install dependencies: `pip install streamlit openai nltk`
3. Add your OpenAI API key as an environment variable: `OPENAI_API_KEY`
4. Run: `streamlit run app.py`

## Context

Built as a dissertation project for BSc Computer Science with 
Artificial Intelligence at the University of Sussex (2026).
