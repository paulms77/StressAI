import os
import pandas as pd

DATA_DIR = os.getenv('STRESS_DATA_DIR', 'data/Subjects_processing')
SUBJECT = os.getenv('SUBJECT', 'subject_02')

src = os.path.join(DATA_DIR, SUBJECT, 'Subject_data.csv')
dst = os.path.join(DATA_DIR, SUBJECT, 'Subject_data_inference_process.csv')

df = pd.read_csv(src)
# df = df.drop(['labels'], axis = 1)

df.to_csv(dst, index = False)
