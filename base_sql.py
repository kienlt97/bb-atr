import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

db = os.environ.get("DB")
# check if there is an database string in the .env file
if db:
    db = db
else:
    # if no variable is specified, use the default string below
    db = "postgresql+psycopg2://postgres:postgres123@192.168.40.11:5432/bot_crypto"

engine = create_engine(db)
engine.connect()

pprint(f"connection successful! : {engine}")
# create a session variable. Allows all our transactions to be ran in the context of a session
Session = sessionmaker(bind=engine)

Base = declarative_base()
