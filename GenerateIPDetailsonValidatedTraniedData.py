import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import threading
from sklearn.metrics import classification_report
import re

class LogProcessor:
    def __init__(self):
        # Configure logging
        logging.basicConfig(filename='script_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.lock = threading.Lock()

    def load_model(self, model_filename):
        try:
            self.best_model = joblib.load(model_filename)
            logging.info("Model loaded successfully.")
            print("Model loaded successfully.")
        except Exception as e:
            logging.error("Error loading the model: %s", str(e))
            print("Error loading the model:", str(e))

    def load_test_data(self, test_data_filename):
        try:
            self.test_df = pd.read_csv(test_data_filename)
            self.test_df.fillna('', inplace=True)
            self.X_test = self.test_df['breach_details']
            self.y_test = self.test_df['attackProbability']
            logging.info("Test data loaded successfully.")
            print("Test data loaded successfully.")
        except Exception as e:
            logging.error("Error loading test data: %s", str(e))
            print("Error loading test data:", str(e))

    def load_vectorizer(self, vectorizer_filename):
        try:
            self.tfidf_vectorizer = joblib.load(vectorizer_filename)
            logging.info("Vectorizer loaded successfully.")
            print("Vectorizer loaded successfully.")
        except Exception as e:
            logging.error("Error loading vectorizer: %s", str(e))
            print("Error loading vectorizer:", str(e))

    def transform_data(self):
        try:
            with self.lock:
                self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
            logging.info("Data transformed successfully.")
            print("Data transformed successfully.")
        except Exception as e:
            logging.error("Error transforming data: %s", str(e))
            print("Error transforming data:", str(e))

    def predict_labels(self):
        try:
            with self.lock:
                self.y_pred_test = self.best_model.predict(self.X_test_tfidf)
            logging.info("Labels predicted successfully.")
            print("Labels predicted successfully.")
        except Exception as e:
            logging.error("Error predicting labels: %s", str(e))
            print("Error predicting labels:", str(e))

    def process_logs(self, model_filename, test_data_filename, vectorizer_filename, validation_excel_filename):
        self.load_model(model_filename)
        self.load_test_data(test_data_filename)
        self.load_vectorizer(vectorizer_filename)

        if all(hasattr(self, attr) for attr in ['best_model', 'X_test', 'tfidf_vectorizer']):
            transform_thread = threading.Thread(target=self.transform_data)
            predict_thread = threading.Thread(target=self.predict_labels)

            transform_thread.start()
            predict_thread.start()

            transform_thread.join()
            predict_thread.join()

            self.test_df['is_attacker'] = ['not_attacker' if is_whitelisted else pred
                                            for is_whitelisted, pred in zip(self.test_df['is_whitelisted'], self.y_pred_test)]

            self.generate_validation_excel(validation_excel_filename)
            self.generate_classification_report('test_classification_report.xlsx')

        else:
            logging.error("Missing required attributes for data processing.")
            print("Missing required attributes for data processing.")

    def generate_validation_excel(self, validation_excel_filename):
        try:
            self.test_df.to_excel(validation_excel_filename, index=False)
            logging.info("Validation data saved to %s", validation_excel_filename)
            print("Validation data saved to", validation_excel_filename)
        except Exception as e:
            logging.error("Error generating validation Excel file: %s", str(e))
            print("Error generating validation Excel file:", str(e))

    def generate_classification_report(self, classification_report_excel_filename):
        try:
            classification_rep = classification_report(self.y_test, self.y_pred_test, output_dict=True)
            classification_report_df = pd.DataFrame(classification_rep).transpose()
            classification_report_df.to_excel(classification_report_excel_filename, index=True)
            logging.info("Classification report saved to %s", classification_report_excel_filename)
            print("Classification report saved to", classification_report_excel_filename)
        except Exception as e:
            logging.error("Error generating classification report: %s", str(e))
            print("Error generating classification report:", str(e))

    def create_ip_blacklist(self, blacklist_filename='ip_blacklist.txt'):
            try:
                # Filter for rows where 'is_attacker' is 'attacker' and get unique IP addresses
                unique_attacker_ips = self.test_df[self.test_df['is_attacker'] == 'attacker']['ip_address'].unique()
                
                # Append '/32' to each IP address
                ip_blacklist_content = "\n".join(ip + '/32' for ip in unique_attacker_ips)
                
                # Write the modified IP addresses to the text file
                with open(blacklist_filename, 'w') as file:
                    file.write(ip_blacklist_content)
                
                logging.info("Unique IP blacklist with /32 prefix created successfully.")
                print("Unique IP blacklist with /32 prefix created successfully.")
            except Exception as e:
                logging.error("Error creating unique IP blacklist with /32 prefix: %s", str(e))
                print("Error creating unique IP blacklist with /32 prefix:", str(e))


if __name__ == "__main__":
    log_processor = LogProcessor()
    log_processor.process_logs('best_model.pkl', 'test_data.csv', 'tfidf_vectorizer.pkl', 'validation_data.xlsx')
    log_processor.create_ip_blacklist('ip_blacklist.txt')
