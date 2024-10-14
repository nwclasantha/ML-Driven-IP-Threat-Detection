import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE for handling class imbalance
import logging
import pickle
import pymysql

class TextClassifier:
    def __init__(self, train_csv_file_path):
        self.train_csv_file_path = train_csv_file_path
        self.logger = logging.getLogger('TextClassifier')
        self._setup_logging()

    def _setup_logging(self):
        # Configure logging
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Log to a file
        file_handler = logging.FileHandler('classification.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Log to console (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_data(self):
        try:
            self.logger.info("Loading training data...")
            train_df = pd.read_csv(self.train_csv_file_path)
            train_df.fillna('', inplace=True)
            self.X = train_df[['region', 'ip_address', 'total_reports', 'domain', 'usage_type', 'isp', 'abuse_confidence_score', 'is_whitelisted', 'breach_details']].values
            self.y = train_df['attackProbability'].values
            self.logger.info(f"Number of samples in training data: {len(train_df)}")
            self.logger.info("Training data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")

    def split_data(self):
        try:
            self.logger.info("Splitting data into training and validation sets...")
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)
            self.logger.info("Data split successfully.")
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")

    def create_tfidf_vectorizer(self):
        self.logger.info("Creating TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer()

    def transform_data(self):
        try:
            self.logger.info("Transforming data using TF-IDF vectorizer...")
            self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train[:, 0])
            self.X_val_tfidf = self.tfidf_vectorizer.transform(self.X_val[:, 0])
            self.logger.info("Data transformation completed.")
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")

    def train_model(self):
        try:
            self.logger.info("Random Forest Training the Model...")

            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(self.X_train_tfidf, self.y_train)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_res, y_train_res)
            self.best_model = model  # Store the trained model
            self.logger.info("Model training completed.")
        except Exception as e:
            self.logger.error(f"Error training the model: {str(e)}")

    def save_model_to_db(self):
        try:
            self.logger.info("Saving the model and TF-IDF vectorizer to the database...")

            # Serialize model and vectorizer
            serialized_model = pickle.dumps(self.best_model)
            serialized_vectorizer = pickle.dumps(self.tfidf_vectorizer)

            # Database connection
            connection = pymysql.connect(host='localhost', user='root', password='0a|xNsl3DZV0a|xZwiu]', db='abusevalidation')
            cursor = connection.cursor()

            # Insert into database
            insert_query = "INSERT INTO models (model_name, model_data) VALUES (%s, %s)"
            cursor.execute(insert_query, ('best_model', serialized_model))
            cursor.execute(insert_query, ('tfidf_vectorizer', serialized_vectorizer))
            
            connection.commit()
            cursor.close()
            connection.close()

            self.logger.info("Model and vectorizer successfully saved to the database.")
        except Exception as e:
            self.logger.error(f"Error saving model and vectorizer to database: {str(e)}")

    def evaluate_model(self):
        try:
            self.logger.info("Evaluating the model on the validation set...")
            y_pred_val = self.best_model.predict(self.X_val_tfidf)
            
            # Adjust target_names as per your classes
            unique_classes = sorted(set(self.y_val))
            target_names = [f'class_{cls}' for cls in unique_classes]

            report = classification_report(self.y_val, y_pred_val, target_names=target_names, zero_division=1)
            print("Classification Report on Validation Set:")
            print(report)
            self.logger.info("Model evaluation completed.")
        except Exception as e:
            self.logger.error(f"Error evaluating the model: {str(e)}")

def main():
    train_csv_file_path = 'train_data.csv'  # Replace with your actual file path
    classifier = TextClassifier(train_csv_file_path)
    classifier.load_data()
    classifier.split_data()
    classifier.create_tfidf_vectorizer()
    classifier.transform_data()
    classifier.train_model()
    classifier.save_model_to_db()  # Saving model to DB instead of file
    classifier.evaluate_model()

if __name__ == "__main__":
    main()
