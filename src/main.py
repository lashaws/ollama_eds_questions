import os
import requests
import json
import logging
import chromadb
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf_processor import extract_text_from_pages


# Setup logging configuration
logging.basicConfig(filename='logs/processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_chromadb():
    client = chromadb.Client()
    return client.get_or_create_collection(name="medical_research")

def process_pdf(file_path, num_pages):
    logging.info(f"Processing file: {file_path}")
    text = extract_text_from_pages(file_path, num_pages)
    return text

def save_questions_to_json(questions, file_path):
    """Saves the generated questions to a JSON file."""
    output_dir = '/Users/lsalta/Dev/llm_hackathon/eds_data/training_data/json_files/'
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(file_path)
    json_filename = f"{base_filename}.json"
    full_path = os.path.join(output_dir, json_filename)

    with open(full_path, 'w') as f:
        json.dump(questions, f, indent=4)

    logging.info(f"Saved questions to {full_path}")

def generate_questions(text, collection, file_id, number_of_questions=2):
    """
    Generates questions based on the provided text using Llama3,
    enhanced by contextual data retrieved from ChromaDB.
    """
    # Insert the document into ChromaDB with a unique ID
    collection.upsert(documents=[text], ids=[file_id])

    # Query for similar documents and prepare the context
    similar_docs = collection.query(query_texts=[text], n_results=5)

    # Generate a combined context from similar documents
    try:
        # Flatten the list of lists of documents to a single list of texts
        all_documents = [doc for sublist in similar_docs['documents'] for doc in sublist]
        combined_context = "\n".join(all_documents)
    except TypeError as e:
        logging.error(f"Failed to process document data from ChromaDB: {e}")
        return "Failed to retrieve similar documents."

    prompt = f"""
    You are tasked with creating a high-quality Q&A dataset based on the provided input text. Your goal is to generate {number_of_questions} questions and their corresponding answers, along with relevant passages from the text and standardized categories.

    <input_text>
    Current File Context: {text}
    Related Information:{combined_context}
    </input_text>

    To create the Q&A dataset, follow these steps:

    1. Carefully read through the input text and identify passages that are suitable for generating questions. Ensure that the selected passages are substantial enough to provide context for the questions and answers.
    2. Create {number_of_questions} questions based on the selected passages. Each question should be clear, concise, and relevant to the corresponding passage.
    3. For each question, provide a detailed and descriptive answer based on the information contained within the relevant passage. Avoid copying the exact passage as the answer; instead, summarize and elaborate on the key points.
    4. Assign a standardized category to each question-answer pair. The category should accurately reflect the main topic or theme of the question and answer. Use your best judgment to create consistent and meaningful categories across the dataset.
    5. Format your output as a list of JSON objects, with each object representing a single question-answer pair. The JSON object should have the following structure:

    {{
      instruction: <created_question>,
      input: <reference_passage_from_text>,
      output: <detailed_answer>
    }}

    Ensure that the "instruction" field contains the generated question, the "input" field contains the relevant passage from the text, the "output" field contains the detailed answer to the question, and the "category" field contains your best guess at a standardized category for the question-answer pair.

    {{
      "instruction": "What is the main theme of the poem 'The Road Not Taken' by Robert Frost?",
      "input": "Two roads diverged in a yellow wood, And sorry I could not travel both And be one traveler, long I stood And looked down one as far as I could To where it bent in the undergrowth; Then took the other, as just as fair, And having perhaps the better claim, Because it was grassy and wanted wear; Though as for that the passing there Had worn them really about the same, And both that morning equally lay In leaves no step had trodden black. Oh, I kept the first for another day! Yet knowing how way leads on to way, I doubted if I should ever come back. I shall be telling this with a sigh Somewhere ages and ages hence: Two roads diverged in a wood, and I— I took the one less traveled by, And that has made all the difference.",
      "output": "The main theme of 'The Road Not Taken' by Robert Frost is the significance of the choices we make in life. The poem presents a metaphorical fork in the road, where the speaker must decide between two paths. The paths represent the different decisions and directions one can take in life. The speaker's choice to take the road 'less traveled by' symbolizes the idea of making unconventional or less popular choices. The poem suggests that these choices, even if they seem small at the time, can have a profound impact on one's life and shape their destiny. Frost emphasizes the importance of individual decision-making and the consequences that follow. The poem also touches on themes of regret, wondering about the path not taken, and the impossibility of knowing where each choice may lead. Ultimately, 'The Road Not Taken' encourages readers to embrace their choices and the unique path they create for themselves."
    }}

    Remember to maintain consistency in your categories and ensure that each question-answer pair is relevant and informative. Your goal is to create a high-quality Q&A dataset that can be used for various applications, such as training language models or creating educational resources.

    Provide your output as a list of JSON objects, following the specified format. NO TEXT BEFORE OR AFTER THE JSON. Provide only the JSON.  
    """

    api_url = "http://localhost:11434/api/chat"
    headers = {'Content-Type': 'application/json'}
    data = {
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ],
        "model": "llama3",
        "stream": False
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an exception for HTTP error codes
        response_data = response.json()
        return response_data['message']['content'] if 'message' in response_data else "No questions generated."
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return "Question generation failed."

def main(pdf_directory, num_pages):
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    collection = setup_chromadb()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, file_path, num_pages): file_path for file_path in pdf_files}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                text = future.result()
                if text:
                    file_id = os.path.basename(file_path)  # Using the file name as a unique ID
                    questions = generate_questions(text, collection, file_id)
                    save_questions_to_json(questions, file_path)
                    logging.info(f"Generated questions for {file_path}: {questions}")
            except Exception as e:
                logging.error(f"File processing failed for {file_path}: {e}")

if __name__ == "__main__":
    main('data/pdfs', num_pages=2)
