import pandas as pd
import spacy
import multiprocessing
import logging

class LogProcessor:
    def __init__(self):
        logging.basicConfig(filename='tokenization.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error loading data from {file_path}. Error: {str(e)}")
            raise

    def tokenize_entry(self, entries, nlp):
        tokenized = []
        for entry in entries:
            try:
                doc = nlp(entry)
                tokens = [token.text for token in doc if not token.is_space]
                tokenized.append(' '.join(tokens))
            except Exception as e:
                logging.error(f"Error tokenizing entry: {entry}. Error: {str(e)}")
                tokenized.append('')
        return tokenized

    def tokenize_columns(self):
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            text_columns = self.df.select_dtypes(include=['object']).columns

            for column in text_columns:
                entries = self.df[column].astype(str).tolist()
                chunks = [entries[i:i + 1000] for i in range(0, len(entries), 1000)]

                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    tokenized_chunks = pool.starmap(self.tokenize_entry, [(chunk, nlp) for chunk in chunks])

                tokenized_entries = [token for sublist in tokenized_chunks for token in sublist]
                self.df[column] = tokenized_entries
        except Exception as e:
            logging.error(f"An error occurred during tokenization: {str(e)}")
            raise

    def save_tokenized_data(self, output_file_path):
        try:
            self.df.to_csv(output_file_path, index=False)
            print(f"Tokenized data saved to {output_file_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving the tokenized data: {str(e)}")
            raise

if __name__ == '__main__':
    try:
        log_processor = LogProcessor()
        cleaned_csv_file_path = 'cleaned_csv_file.csv'
        log_processor.load_data(cleaned_csv_file_path)
        log_processor.tokenize_columns()
        tokenized_csv_file_path = 'tokenized_csv_file.csv'
        log_processor.save_tokenized_data(tokenized_csv_file_path)
    except Exception as e:
        logging.error(f"An error occurred in the main process: {str(e)}")
