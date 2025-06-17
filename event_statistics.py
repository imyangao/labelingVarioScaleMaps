import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Database connection settings
db_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': '',
    'user': '',
    'password': ''
}

# SQL query
query = """
WITH steps AS (
  SELECT DISTINCT face_id, step_value
  FROM label_anchors
)
SELECT face_id,
       COUNT(*) AS key_step_count
FROM steps
GROUP BY face_id
ORDER BY key_step_count DESC;
"""

# Connect to the database and fetch results
try:
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql_query(query, conn)
    conn.close()
except Exception as e:
    print("Error connecting to database or executing query:", e)
    exit()

# Optional: limit the number of face_ids shown (for readability)
df = df.head(50)  # adjust this number if needed

# Plotting bar chart
plt.figure(figsize=(14, 6))
plt.bar(df['face_id'].astype(str), df['key_step_count'])
plt.xticks(rotation=90, fontsize=14)
plt.title('Number of Key Steps per Face ID', fontsize=16)
plt.xlabel('Face ID', fontsize=16)
plt.ylabel('Key Step Count', fontsize=16)
plt.tight_layout()
plt.grid(axis='y')
plt.show()