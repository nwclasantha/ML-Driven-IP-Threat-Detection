import pandas as pd
import re
from multiprocessing import Pool
import logging

class LogCleaner:
    def __init__(self, input_csv_path, output_csv_path, num_processes=4):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.num_processes = num_processes

    def clean_log_entry(self, log_entry):
        if isinstance(log_entry, str):
            try:
                # Adjust the cleaning logic as per the actual data in the CSV file
                # Example: Removing IP addresses from domain field
                log_entry = re.sub(r'\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', '', log_entry)
                return log_entry.strip()
            except Exception as e:
                # Log the exception
                logging.error(f"Error cleaning log entry: {e}")
                return ''
        else:
            return ''

    def clean_chunk(self, chunk):
        # Adjusting to the correct column names from the CSV file
        # For example, if we are cleaning 'domain' column
        chunk['domain'] = chunk['domain'].apply(self.clean_log_entry)
        return chunk

    def process_logs(self):
        # Configure the logging
        logging.basicConfig(filename='log_cleaning.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load the CSV file containing log entries
        df = pd.read_csv(self.input_csv_path)

        if len(df) == 0:
            logging.warning("Input DataFrame is empty. No cleaning required.")
            return  # Exit early if there are no log entries

        # Calculate the chunk size, ensuring it's not zero
        chunk_size = max(len(df) // self.num_processes, 1)

        # Split the DataFrame into chunks for parallel processing
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Create a Pool of worker processes
        with Pool(processes=self.num_processes) as pool:
            try:
                cleaned_chunks = pool.map(self.clean_chunk, chunks)
            except Exception as e:
                # Log the exception
                logging.error(f"Error during concurrent processing: {e}")
                raise  # Re-raise the exception

        # Concatenate the cleaned DataFrame chunks back together
        df = pd.concat(cleaned_chunks, ignore_index=True)

        # Save the cleaned data to a new CSV file
        df.to_csv(self.output_csv_path, index=False)

        print(f"Cleaned data saved to {self.output_csv_path}")

if __name__ == '__main__':
    input_csv_file = 'labeled_logs.csv'
    output_csv_file = 'cleaned_csv_file.csv'
    num_processes = 4

    log_cleaner = LogCleaner(input_csv_file, output_csv_file, num_processes)
    log_cleaner.process_logs()
