import mysql.connector
import csv
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Subclass for processing rows
class RowProcessor:
    @staticmethod
    def process(row):
        # Extract the relevant fields from the row
        is_whitelisted, abuse_confidence_score = row[7], row[6]
        # Determine the attackProbability based on the conditions
        attack_probability = "attacker" if is_whitelisted == 0 else "not_attacker"
        # Return the updated row
        return list(row) + [attack_probability]

# Main class for exporting data to CSV
class DatabaseCSVExporter:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.host, 
            database=self.database, 
            user=self.user, 
            password=self.password
        )
        return self.conn.is_connected()

    def export_to_csv(self, filename):
        try:
            if self.connect():
                logging.info('Connected to MySQL database')

                query = """
                SELECT region, ip_address, total_reports, domain, usage_type, isp, 
                       abuse_confidence_score, is_whitelisted, breach_details
                FROM validated_data                 
                """
                cursor = self.conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                logging.info(f"Number of rows fetched: {len(rows)}")

                if len(rows) > 0:
                    # Get column names from the cursor description
                    column_names = [i[0] for i in cursor.description] + ['attackProbability']

                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(column_names)  # Write the header

                        with ThreadPoolExecutor() as executor:
                            # Process each row in parallel
                            for processed_row in executor.map(RowProcessor.process, rows):
                                writer.writerow(processed_row)  # Write the processed row to the CSV

                    logging.info("Data exported to CSV successfully")
                else:
                    logging.info("No data to export")

                cursor.close()
            else:
                logging.error("Failed to connect to the database")
            self.conn.close()

        except mysql.connector.Error as err:
            logging.error(f"Error in MySQL connection: {err}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

# Usage
exporter = DatabaseCSVExporter("localhost", "abusevalidation",  "xxxxxxxxxx", "xxxxxxxxxxx")
exporter.export_to_csv('labeled_logs.csv')
