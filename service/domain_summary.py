import concurrent.futures
import json
import logging
import os

import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI

# Load configuration
with open("service/config.json", "r") as f:
    config = json.load(f)

# Set up the OpenAI client
client = OpenAI(
    api_key=os.environ.get(
        "OPENAI_API_KEY"
    ),  # Set your OpenAI API key in environment variable
)

# Initialize the tokenizer
tokenizer = tiktoken.encoding_for_model(
    config["openai_model"]
)  # You can use gpt-3.5-turbo or the appropriate model

MAX_TOKENS = config["max_tokens"]


def get_all_links(domain, max_depth=2):
    visited_links = set()

    def is_valid_link(link):
        return domain in link and link not in visited_links

    def prepare_link(href):
        if href.startswith("/"):
            href = domain + href
        return href if is_valid_link(href) else None

    def fetch_anchors_in_page(link):
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, "html.parser")
            anchors = [
                prepare_link(anchor["href"]) for anchor in soup.find_all("a", href=True)
            ]
            return [anchor for anchor in anchors if anchor is not None]
        except Exception as e:
            logging.error(f"Error fetching anchors at {link}: {e}")
            return []

    def fetch_and_process_link(link, current_depth):
        if current_depth >= max_depth or link in visited_links:
            return
        visited_links.add(link)

        next_links = fetch_anchors_in_page(link)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fetch_and_process_link, next_link, current_depth + 1)
                for next_link in next_links
            ]

        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    fetch_and_process_link(domain, 0)

    print(f"Collected links: {visited_links}")

    return visited_links


def get_site_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    text = " ".join(text.split())
    logging.info(f"Collected text from {url}: {text[:200]}...")
    return text


def summarize_text(text, language="sv"):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": config["summary_prompt"][language]},
            {"role": "user", "content": text},
        ],
        model=config["openai_model"],
    )
    summary = response.choices[0].message.content.strip()
    return summary


def chunk_texts(text_list, max_tokens=MAX_TOKENS):
    chunks = []
    current_chunk = []
    current_length = 0

    for text in text_list:
        encoded_length = len(tokenizer.encode(text))
        if current_length + encoded_length + 1 > max_tokens:
            # Add current chunk to chunks list
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(text)
        current_length += encoded_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def summarize_domain(domain):
    links = get_all_links(domain)
    page_texts = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_site_text, link) for link in links]
        for future in concurrent.futures.as_completed(futures):
            page_texts.append(future.result())

    if not page_texts:
        return "No text was collected from the domain."

    chunks = chunk_texts(page_texts)

    # Using a ThreadPoolExecutor to make concurrent API calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_text, chunks))

    combined_summary = " ".join(summaries)
    summary_chunks = chunk_texts([combined_summary], max_tokens=MAX_TOKENS // 2)

    # Using a ThreadPoolExecutor for second round of summarization
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summary_chunks = list(executor.map(summarize_text, summary_chunks))

    final_summary = " ".join(summary_chunks)

    return final_summary
