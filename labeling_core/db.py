import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus

##############################################################################
# Database connection
##############################################################################
DB_NAME = ""
DB_USER = ""
DB_PASS = ""
DB_HOST = "localhost"
DB_PORT = 5432

##############################################################################
# Feature-class ranges
##############################################################################
ROAD_MIN, ROAD_MAX = 10000, 11000
WATER_MIN, WATER_MAX = 12000, 13000
BULD_MIN, BULD_MAX = 13000, 14000


def get_connection(autocommit=True):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = autocommit
    return conn

def get_engine():
    escaped_password = quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{escaped_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_uri) 