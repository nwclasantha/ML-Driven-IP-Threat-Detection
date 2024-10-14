import pandas as pd
import joblib
import logging
import threading
from sklearn.metrics import classification_report
import mysql.connector
from mysql.connector import Error
from io import BytesIO

class LogProcessor:
    def __init__(self, test_data_filename, validation_excel_filename, db_host, db_user, db_password, db_name):
        # Basic logging configuration
        logging.basicConfig(filename='script_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.lock = threading.Lock()

        # Setting up class attributes with the passed parameters
        self.test_data_filename = test_data_filename
        self.validation_excel_filename = validation_excel_filename
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name

    def load_model(self):
        # Connect to the database and load the model
        try:
            connection = mysql.connector.connect(host=self.db_host, user=self.db_user, password=self.db_password, database=self.db_name)
            cursor = connection.cursor(buffered=True)
            cursor.execute("SELECT model_data FROM models WHERE model_name = 'best_model'")
            model_data = cursor.fetchone()
            if model_data:
                model_buffer = BytesIO(model_data[0])
                self.best_model = joblib.load(model_buffer)
                logging.info("Model loaded successfully from the database.")
                print("Model loaded successfully from the database.")
            else:
                logging.error("No model data found in the database.")
                print("No model data found in the database.")
        except Exception as e:
            logging.error("Error loading the model from the database: %s", str(e))
            print("Error loading the model from the database:", str(e))
        finally:
            cursor.close()
            connection.close()

    def load_test_data(self):
        # Load the test data from a CSV file
        try:
            self.test_df = pd.read_csv(self.test_data_filename)
            self.test_df.fillna('', inplace=True)
            self.X_test = self.test_df['breach_details']
            self.y_test = self.test_df['attackProbability']
            logging.info("Test data loaded successfully.")
            print("Test data loaded successfully.")
        except Exception as e:
            logging.error("Error loading test data: %s", str(e))
            print("Error loading test data:", str(e))

    def load_vectorizer(self):
        # Connect to the database and load the vectorizer
        try:
            connection = mysql.connector.connect(host=self.db_host, user=self.db_user, password=self.db_password, database=self.db_name)
            cursor = connection.cursor(buffered=True)
            cursor.execute("SELECT model_data FROM models WHERE model_name = 'tfidf_vectorizer'")
            vectorizer_data = cursor.fetchone()
            if vectorizer_data:
                vectorizer_buffer = BytesIO(vectorizer_data[0])
                self.tfidf_vectorizer = joblib.load(vectorizer_buffer)
                logging.info("Vectorizer loaded successfully from the database.")
                print("Vectorizer loaded successfully from the database.")
            else:
                logging.error("No vectorizer data found in the database.")
                print("No vectorizer data found in the database.")
        except Exception as e:
            logging.error("Error loading vectorizer from the database: %s", str(e))
            print("Error loading vectorizer from the database:", str(e))
        finally:
            cursor.close()
            connection.close()

    def transform_data(self):
        # Transform the test data using the loaded vectorizer
        try:
            with self.lock:
                self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
            logging.info("Data transformed successfully.")
            print("Data transformed successfully.")
        except Exception as e:
            logging.error("Error transforming data: %s", str(e))
            print("Error transforming data:", str(e))

    def predict_labels(self):
        # Use the loaded model to predict labels
        try:
            with self.lock:
                self.y_pred_test = self.best_model.predict(self.X_test_tfidf)
            logging.info("Labels predicted successfully.")
            print("Labels predicted successfully.")
        except Exception as e:
            logging.error("Error predicting labels: %s", str(e))
            print("Error predicting labels:", str(e))

    def process_logs(self):
        # Main method to process logs
        self.load_model()
        self.load_test_data()
        self.load_vectorizer()

        if all(hasattr(self, attr) for attr in ['best_model', 'X_test', 'tfidf_vectorizer']):
            transform_thread = threading.Thread(target=self.transform_data)
            predict_thread = threading.Thread(target=self.predict_labels)

            transform_thread.start()
            predict_thread.start()

            transform_thread.join()
            predict_thread.join()

            self.generate_classification_report()
        else:
            logging.error("Missing required attributes for data processing.")
            print("Missing required attributes for data processing.")

    def generate_classification_report(self):
        # Generate a classification report
        try:
            self.test_df['predicted_label'] = self.test_df.apply(
                lambda row: 'attacker' if (row['is_whitelisted'] == 0 and row['abuse_confidence_score'] > 0) else 'not_attacker', 
                axis=1)

            classification_rep = classification_report(self.y_test, self.test_df['predicted_label'], output_dict=True)
            classification_report_df = pd.DataFrame(classification_rep).transpose()

            classification_report_df.to_excel('test_classification_report.xlsx', index=True)
            logging.info("Classification report saved to test_classification_report.xlsx")
            print("Classification report saved to test_classification_report.xlsx")
        except Exception as e:
            logging.error("Error generating classification report: %s", str(e))
            print("Error generating classification report:", str(e))

    def create_ip_blacklist(self, blacklist_filename):
        # Create an IP blacklist file
        try:
            attacker_ips = self.test_df[self.test_df['predicted_label'] == 'attacker']['ip_address'].unique()
            with open(blacklist_filename, 'w') as file:
                for ip in attacker_ips:
                    file.write(ip + '\n')
            logging.info("IP blacklist created successfully.")
            print("IP blacklist created successfully.")
        except Exception as e:
            logging.error("Error creating IP blacklist: %s", str(e))
            print("Error creating IP blacklist:", str(e))

    def connect_to_database(self):
        # Connect to the MySQL database
        try:
            self.connection = mysql.connector.connect(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                logging.info("Connected to MySQL Server version %s", db_info)
                print("Connected to MySQL Server version ", db_info)
                self.cursor = self.connection.cursor()
        except Error as e:
            logging.error("Error while connecting to MySQL %s", str(e))
            print("Error while connecting to MySQL", str(e))
            self.connection = None

    def insert_data_to_database(self, table_name):
        # Insert data into a specified table in the MySQL database
        try:
            for _, row in self.test_df.iterrows():
                check_query = f"SELECT COUNT(*) FROM {table_name} WHERE ip_address = %s"
                self.cursor.execute(check_query, (row['ip_address'],))
                if self.cursor.fetchone()[0] == 0:
                    sql_query = f"""INSERT INTO {table_name} (region, ip_address, total_reports, domain, usage_type, isp, abuse_confidence_score, is_whitelisted, breach_details, attackProbability)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
                    insert_values = (
                        row['region'], row['ip_address'], row['total_reports'], row['domain'], 
                        row['usage_type'], row['isp'], row['abuse_confidence_score'], row['is_whitelisted'], 
                        row['breach_details'], row['attackProbability']
                    )
                    self.cursor.execute(sql_query, insert_values)
                else:
                    logging.info(f"IP address {row['ip_address']} already exists in {table_name}, skipping insertion.")
            self.connection.commit()
            logging.info("Data inserted successfully into %s table", table_name)
            print("Data inserted successfully into ", table_name)
        except Error as e:
            self.connection.rollback()
            logging.error("Failed to insert data into MySQL table %s", str(e))
            print("Failed to insert data into MySQL table", str(e))
        finally:
            if self.connection.is_connected():
                self.cursor.close()
                self.connection.close()
                logging.info("MySQL connection is closed")
                print("MySQL connection is closed")

if __name__ == "__main__":
    # Instantiate LogProcessor with the provided filenames and database credentials
    log_processor = LogProcessor('test_data.csv', 'test_classification_report.xlsx',  'localhost', 'xxxxxxxx', 'xxxxxxxxxxxx', 'abusevalidation')

    # Process logs
    log_processor.process_logs()


    # If test_df is not created, there's no point in proceeding to the database operations
    if not hasattr(log_processor, 'test_df'):
        logging.error("The test_df attribute has not been set. Exiting.")
        print("The test_df attribute has not been set. Exiting.")
        exit(1)

    # Connect to the database
    log_processor.connect_to_database()

    # Check if the connection is established
    if log_processor.connection is None or not log_processor.connection.is_connected():
        logging.error("Failed to connect to the database. Exiting.")
        print("Failed to connect to the database. Exiting.")
        exit(1)

    # Insert data into the database
    log_processor.insert_data_to_database('ml_validated_data')
