# Medical Research Q&A Dataset Generator

This project is a Python application that processes PDF files, extracts text, generates questions based on the extracted text, and saves the generated questions to a JSON file. It uses the Llama3 model for question generation and ChromaDB for contextual data retrieval.

## Dependencies

The project depends on the following Python libraries:

- `fitz` (PyMuPDF)
- `requests`
- `json`
- `logging`
- `chromadb`
- `concurrent.futures`
- `ollama`

These dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```

## Setting Up a Virtual Environment
To ensure that the ollama package doesn't interfere with other Python projects, it's a good practice to use a virtual environment. Here's how to set it up:
First, create a new virtual environment. You can do this using the venv module that comes with Python. Open your terminal and run:

```bash
python3 -m venv env
```
Activate the virtual environment. The command to do this will depend on your operating system:
On macOS and Linux:
```bash
source env/bin/activate
```
On Windows:
```bash
.\env\Scripts\activate
```

Once the virtual environment is activated, you can install the ollama package locally to this environment:

```bash
pip install ollama
```

Remember to activate the virtual environment every time you work on this project. When you're done, you can deactivate the virtual environment by simply typing deactivate in your terminal.  

### Project Structure
The project consists of the following Python scripts:  
src/main.py: 
- The main script that orchestrates the processing of PDF files, question generation, and saving of questions to JSON files.
- src/pdf_processor.py: A utility script that extracts text from PDF files.
Usage

To run the project, navigate to the project directory and run the main.py script:

```bash
python src/main.py
```
By default, the script processes PDF files located in the data/pdfs directory and generates 2 questions per file. The generated questions are saved to JSON files in the /Users/lsalta/Dev/llm_hackathon/eds_data/training_data/json_files/ directory.  

### Logging

The application logs its activities to a file named processing.log located in the logs directory. The log file includes information about the processing of each PDF file and any errors that occur during the process.  

### Contributing
Contributions are welcome. Please open an issue to discuss your idea or submit a Pull Request with your changes.