from openpyxl import load_workbook
import pymysql
import os
import pandas as pd

db = pymysql.connect(
  host=os.getenv('DB_HOST', 'localhost'),
  user=os.getenv('DB_USER', 'root'),
  password=os.getenv('DB_PASSWORD', ''),
  database=os.getenv('DB_NAME', 'stress_db'),
  charset='utf8mb4'
)

cursor = db.cursor()

data_dir = os.getenv('STRESS_DATA_DIR', 'data/Subjects_processing')

for subject_folder in sorted(os.listdir(data_dir)):
  subject_path = os.path.join(data_dir, subject_folder)

  if os.path.isdir(subject_path) and subject_folder.startswith('subject_'):
    print(subject_folder)

    cursor.execute('INSERT IGNORE INTO subjects (subject_name) VALUES (%s)', (subject_folder,))
    db.commit()

    cursor.execute('SELECT id FROM subjects WHERE subject_name = %s', (subject_folder,))
    subject_id = cursor.fetchone()[0]
    print(subject_id)

    subject_file = os.path.join(subject_path, 'Subject_data.csv')
    if os.path.exists(subject_file):
      subject_data = pd.read_csv(subject_file)
      for _, row in subject_data.iterrows():
        cursor.execute('INSERT INTO eda_data (subject_id, eda) VALUES (%s, %s)', (subject_id, row['eda']))
        cursor.execute('INSERT INTO bvp_data (subject_id, bvp) VALUES (%s, %s)', (subject_id, row['bvp']))
        cursor.execute('INSERT INTO label_data (subject_id, labels) VALUES (%s, %s)', (subject_id, row['labels']))

        db.commit()
       
cursor.close()
db.close()

print('All Subjects Saved in MySQL DataBase.')