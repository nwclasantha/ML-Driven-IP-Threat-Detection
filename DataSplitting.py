import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import concurrent.futures

class DataProcessor:
    def __init__(self, tokenized_csv_file_path, train_csv_file_path, test_csv_file_path, test_size=0.2, random_seed=42):
        self.tokenized_csv_file_path = tokenized_csv_file_path
        self.train_csv_file_path = train_csv_file_path
        self.test_csv_file_path = test_csv_file_path
        self.test_size = test_size
        self.random_seed = random_seed

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Log to a file
        file_handler = logging.FileHandler('data_processor.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Log to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_data(self):
        try:
            # Load the CSV file containing tokenized log entries and labels
            self.df = pd.read_csv(self.tokenized_csv_file_path)
        except Exception as e:
            self._log_error("An error occurred while loading data:", e)
            raise

    def split_data(self):
        try:
            # Update these column names based on your tokenized CSV file's structure
            features = ['region', 'ip_address', 'total_reports', 'domain', 'usage_type', 'isp', 'abuse_confidence_score', 'is_whitelisted', 'breach_details']
            label = 'attackProbability'  # Replace with your actual label column name

            X = self.df[features]
            y = self.df[label]

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)

            self.train_df = pd.concat([X_train, y_train], axis=1)
            self.test_df = pd.concat([X_test, y_test], axis=1)
        except Exception as e:
            self._log_error("An error occurred while splitting data:", e)
            raise

    def save_data(self):
        try:
            # Save the training and testing sets to separate CSV files concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._save_csv, self.train_df, self.train_csv_file_path),
                           executor.submit(self._save_csv, self.test_df, self.test_csv_file_path)]

                # Wait for both tasks to complete
                concurrent.futures.wait(futures)

            self.logger.info(f"Training data saved to {self.train_csv_file_path}")
            self.logger.info(f"Testing data saved to {self.test_csv_file_path}")
        except Exception as e:
            self._log_error("An error occurred while saving data:", e)
            raise

    def _save_csv(self, df, file_path):
        df.to_csv(file_path, index=False)

    def _log_error(self, message, error):
        # Log error messages
        self.logger.error(f"{message} {str(error)}")

if __name__ == "__main__":
    # Define file paths and parameters
    tokenized_csv_file_path = 'tokenized_csv_file.csv'
    train_csv_file_path = 'train_data.csv'
    test_csv_file_path = 'test_data.csv'
    test_size = 0.2
    random_seed = 42

    # Create a DataProcessor instance
    data_processor = DataProcessor(tokenized_csv_file_path, train_csv_file_path, test_csv_file_path, test_size, random_seed)

    # Load, split, and save data
    data_processor.load_data()
    data_processor.split_data()
    data_processor.save_data()
