import base64
import csv
import os
import re
from datetime import datetime

import nltk
import streamlit as st
from openai import OpenAI
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

# Basic setup for where the app is running from.
# This helps keep downloaded NLTK files inside the project folder.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_NLTK_DATA = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(LOCAL_NLTK_DATA, exist_ok=True)

# Force NLTK to use only this local folder
nltk.data.path = [LOCAL_NLTK_DATA]

# Makes sure the NLTK files needed for tokenising and POS tagging are available.
# This avoids the app failing if the files have not been downloaded yet.
def ensure_nltk_data():
    required_paths = [
        (os.path.join(LOCAL_NLTK_DATA, "tokenizers", "punkt"), "punkt"),
        (os.path.join(LOCAL_NLTK_DATA, "tokenizers", "punkt_tab"), "punkt_tab"),
        (os.path.join(LOCAL_NLTK_DATA, "taggers", "averaged_perceptron_tagger_eng"), "averaged_perceptron_tagger_eng"),
    ]

    for local_path, download_name in required_paths:
        if not os.path.exists(local_path):
            nltk.download(download_name, download_dir=LOCAL_NLTK_DATA, quiet=False)


ensure_nltk_data()

# OpenAI client used for the image generation part.
client = OpenAI()

# Sends the prepared prompt to the image model.
# Returning None here lets the interface show a warning instead of crashing.
def generate_image(prompt: str):
    try:
        result = client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            size="1024x1024"
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        return image_bytes
    except Exception:
        return None

# Changes the image prompt depending on the writing score.
# This is what makes weaker writing produce simpler images.
def build_image_prompt(text: str, score: int):
    if score < 30:
        style_instruction = (
            "Create a very poor-quality, child-friendly drawing of the scene. "
            "Make it look unfinished, unclear, and minimally helpful. "
            "Use only black and white. "
            "No shading, no realistic detail, no rich textures, no colourful elements. "
            "Use very simple stick-figure or basic shape style where possible. "
            "Keep the background plain and mostly empty. "
            "The scene should feel basic, flat, and low-detail, like a rough first sketch. "
            f"Scene description: {text}"
        )
        return style_instruction

    elif score < 60:
        style_instruction = (
            "Create a simple but clear child-friendly illustration of the scene. "
            "Include some colour, a moderate amount of detail but not good looking. "
            "The image should be understandable but not highly detailed or visually rich. "
            f"Scene description: {text}"
        )
        return style_instruction

    else:
        return text

# Extra descriptive words used to reward sensory and visual detail in the writing.
DESCRIPTIVE_WORDS = {
    "bright", "dark", "shiny", "dull", "glowing", "sparkling", "golden", "silver",
    "colourful", "colorful", "pale", "shadowy", "glittering", "faint", "radiant",
    "cold", "warm", "hot", "freezing", "icy", "chilly", "cool", "burning", "humid",
    "loud", "quiet", "silent", "whisper", "whispering", "scream", "screaming",
    "echoing", "rustling", "crackling", "buzzing", "roaring", "humming",
    "sweet", "bitter", "sour", "salty", "spicy", "sugary",
    "rough", "smooth", "soft", "hard", "slippery", "sticky", "silky", "fuzzy",
    "sharp", "gentle", "bumpy", "wet", "dry", "dusty", "muddy", "damp",
    "smell", "scent", "stink", "fragrant", "perfume", "musty", "fresh", "smoky",
    "foggy", "misty", "windy", "rainy", "stormy", "cloudy", "sunny", "snowy",
    "eerie", "calm", "gloomy", "peaceful", "wild", "ancient", "enormous", "tiny",
    "massive", "little", "beautiful", "ugly", "lovely", "graceful", "gentle",
    "glimmering", "twinkling", "crisp", "hazy", "murky", "glassy", "frozen"
}

# NLTK adjective tags used when counting describing words.
ADJECTIVE_TAGS = {"JJ", "JJR", "JJS"}

# Simple word lists for the safety filter
# These are not perfect but they help block obvious unsafe classroom content.
SEXUAL_WORDS = {
    "sex", "sexy", "nude", "naked", "kiss", "kissing", "boobs", "breasts",
    "penis", "vagina", "porn", "bedroom"
}

VIOLENCE_WORDS = {
    "kill", "killing", "murder", "blood", "bloody", "stab", "stabbing",
    "gun", "knife", "dead", "death", "hurt", "hanging", "corpse"
}

HATE_WORDS = {
    "racist", "hate", "slur"
}

SELF_HARM_WORDS = {
    "suicide", "self-harm", "selfharm", "cutting"
}

DRUG_WORDS = {
    "drugs", "cocaine", "weed", "marijuana", "smoking", "vape", "alcohol", "beer", "vodka"
}

# Cleans words so the safety checks are easier to match.
def clean_word(word: str):
    return "".join(char.lower() for char in word if char.isalpha())


# Checks for personal details that should not be submitted by a child user.
def contains_personal_info(text: str):
    lower_text = text.lower()

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    phone_pattern = r"\b(?:\+?\d[\d\s\-]{7,}\d)\b"

    if re.search(email_pattern, text):
        return True, "personal_info"

    if re.search(phone_pattern, text):
        return True, "personal_info"

    personal_phrases = [
        "my phone number is",
        "my email is",
        "my address is",
        "i live at",
        "my school is",
        "my name is",
        "my full name is",
    ]

    for phrase in personal_phrases:
        if phrase in lower_text:
            return True, "personal_info"

    return False, None


# Runs safety checks before the text is scored or sent for image generation.
def check_safety(text: str):
    lower_text = text.lower()
    raw_tokens = word_tokenize(lower_text)
    cleaned_tokens = {clean_word(token) for token in raw_tokens if clean_word(token)}

# Some unsafe phrases need to be checked as full phrases, not single words.
    multiword_checks = [
        ("kill myself", "self_harm"),
        ("hurt myself", "self_harm")
    ]

    for phrase, category in multiword_checks:
        if phrase in lower_text:
            return {
                "safe": False,
                "category": category,
                "message": "Please change your writing so it stays safe and classroom-friendly."
            }

    personal_info_found, personal_info_category = contains_personal_info(text)
    if personal_info_found:
        return {
            "safe": False,
            "category": personal_info_category,
            "message": "Please remove personal details like names, email addresses, phone numbers, or school information."
        }

    category_checks = [
        ("sexual_content", SEXUAL_WORDS),
        ("violence", VIOLENCE_WORDS),
        ("hate", HATE_WORDS),
        ("self_harm", SELF_HARM_WORDS),
        ("drugs_alcohol", DRUG_WORDS)
    ]

    for category, word_set in category_checks:
        if cleaned_tokens.intersection(word_set):
            return {
                "safe": False,
                "category": category,
                "message": "Please change your writing so it stays safe and classroom-friendly."
            }

    return {
        "safe": True,
        "category": None,
        "message": None
    }

# Calculates the writing score from simple features that can be measured.
def richness_score(text: str):
    t = text.strip()

    if t == "":
        return 0, {
            "words": 0,
            "unique_words": 0,
            "sentences": 0,
            "adjectives": 0,
            "detail_words": 0
        }

# Keep word-like tokens only, so punctuation does not affect the word count.
    raw_tokens = word_tokenize(t)
    words = [token.lower() for token in raw_tokens if any(char.isalpha() for char in token)]

    word_count = len(words)
    unique_words = len(set(words))

    sentences = sent_tokenize(t)
    sentence_count = len(sentences)

# POS tagging is used here to count adjectives as describing words
    tagged_words = pos_tag(words)
    adjective_count = sum(1 for _, tag in tagged_words if tag in ADJECTIVE_TAGS)

    detail_word_hits = sum(1 for word in words if word in DESCRIPTIVE_WORDS)

# Caps are used so one feature cannot push the score too high by itself
    score = 0
    score += min(word_count, 50) * 0.5
    score += min(unique_words, 25) * 0.8
    score += min(sentence_count, 3) * 5.0
    score += min(adjective_count, 10) * 2.0
    score += min(detail_word_hits, 10) * 2.0

    score = int(min(score, 100))

    breakdown = {
        "words": word_count,
        "unique_words": unique_words,
        "sentences": sentence_count,
        "adjectives": adjective_count,
        "detail_words": detail_word_hits
    }

    return score, breakdown

# Turns the score breakdown into simple feedback messages.
def make_feedback(breakdown: dict):
    messages = []

    words = breakdown.get("words", 0)
    unique_words = breakdown.get("unique_words", 0)
    sentences = breakdown.get("sentences", 0)
    adjectives = breakdown.get("adjectives", 0)
    detail_words = breakdown.get("detail_words", 0)

    if words < 20:
        messages.append("Try writing a bit more ✍️ Add 1–2 more ideas about what is happening.")
    elif words < 50:
        messages.append("Good length ✅ Now add more detail to make it clearer.")
    else:
        messages.append("Great amount of writing 🌟")

    if sentences <= 1:
        messages.append("Try using more sentences 🧠 Split your ideas using full stops.")
    elif sentences <= 3:
        messages.append("Nice! ✅ Try adding one more sentence to build the scene.")
    else:
        messages.append("Great sentence variety 🌟")

# This checks whether the writing repeats the same words too much.
    ratio = (unique_words / words) if words > 0 else 0
    if ratio < 0.45:
        messages.append("Try using a few different words 🔁 Swap repeated words for new ones.")
    else:
        messages.append("Nice vocabulary ✅")

    if adjectives == 0:
        messages.append("Add more describing words 🎨 Try using adjectives like dark, cold, huge, or shiny.")
    elif adjectives < 3:
        messages.append("Good start with describing words ✅ Try adding a few more adjectives.")
    else:
        messages.append("Great use of describing words 🌟")

    if detail_words == 0:
        messages.append("Add sensory details 👀👂 What can you see, hear, smell, taste, or feel?")
    elif detail_words < 3:
        messages.append("Good start on detail ✅ Try adding 1–2 more vivid words.")
    else:
        messages.append("Awesome detail 🌟 Your description helps build a clear picture.")

    return messages

# Saves only anonymous attempt data for testing and evaluation.
# The final version does not save the full text written by the user.
def log_attempt_to_csv(
    attempt_number: int,
    score: int,
    breakdown: dict,
    csv_path: str = "attempt_logs.csv"
):
    file_exists = False
    try:
        with open(csv_path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "attempt_number": attempt_number,
        "score": score,
        "words": breakdown.get("words", 0),
        "unique_words": breakdown.get("unique_words", 0),
        "sentences": breakdown.get("sentences", 0),
        "adjectives": breakdown.get("adjectives", 0),
        "detail_words": breakdown.get("detail_words", 0),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Main app interface
st.title("Descriptive Writing Visualiser")
st.write("Write a description and click Generate. Improve your descriptive writing and see it come to life through AI-generated images. You can revise your writing and try again based on the feedback provided.")

if "attempt_number" not in st.session_state:
    st.session_state.attempt_number = 0

if "last_score" not in st.session_state:
    st.session_state.last_score = None

if "last_breakdown" not in st.session_state:
    st.session_state.last_breakdown = None

if "last_text" not in st.session_state:
    st.session_state.last_text = None

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Attempt number:** {st.session_state.attempt_number + 1}")

with col2:
    # Reset clears the progress for a new writing attempt.
    if st.button("Start over (Reset)"):
        st.session_state.attempt_number = 0
        st.session_state.last_score = None
        st.session_state.last_breakdown = None
        st.session_state.last_text = None
        st.rerun()

text = st.text_area("Write your description here:", height=170)

# Main button logic for checking, scoring, feedback, image generation and logging.
if st.button("Generate"):
    if text.strip() == "":
        st.warning("Please write something first.")
    else:
        # Safety is checked first so unsafe text is not scored or sent to the API.
        safety_result = check_safety(text)

        if not safety_result["safe"]:
            st.error(safety_result["message"])
            st.info("Please edit your writing and try again.")
        else:
            st.session_state.attempt_number += 1

# Get the writing score and feedback after the input has passed safety checks.
            score, breakdown = richness_score(text)
            feedback = make_feedback(breakdown)

            try:
                log_attempt_to_csv(st.session_state.attempt_number, score, breakdown)
                log_saved = True
            except PermissionError:
                log_saved = False
                st.warning("Could not save to attempt_logs.csv because the file may be open. Please close it and try again.")
            except Exception:
                log_saved = False
                st.warning("Something went wrong while saving the attempt log.")

            st.subheader("You wrote:")
            st.write(text)

            st.subheader("Richness score:")
            st.write(f"{score} / 100")

            if st.session_state.last_score is None:
                st.info("This is your first attempt. You can improve it and click Generate again.")
            else:
                delta = score - st.session_state.last_score
                if delta > 0:
                    st.success(f"Nice! Your score improved by +{delta}.")
                elif delta < 0:
                    st.warning(f"Your score went down by {delta}. That’s okay — try one tip below.")
                else:
                    st.info("Your score stayed the same. Try one tip below to improve it.")

            st.subheader("Helpful feedback:")
            for msg in feedback:
                st.write("• " + msg)

            st.subheader("Generated image:")

# The score changes how detailed the image prompt should be.
            image_prompt = build_image_prompt(text, score)

            with st.spinner("Generating image..."):
                image_bytes = generate_image(image_prompt)

            if image_bytes is not None:
                st.image(image_bytes)
            else:
                st.warning("Image generation failed. Please try again.")

            st.session_state.last_score = score
            st.session_state.last_breakdown = breakdown
            st.session_state.last_text = text

            if log_saved:
                st.caption("Saved anonymous attempt metrics to attempt_logs.csv (in your project folder).")