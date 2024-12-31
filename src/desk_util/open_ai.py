import json
import logging
import os

from openai import OpenAI
from openai.lib.azure import AzureOpenAI

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import data_root_path
from rule_gen.cpath import output_root_path


def get_state_db_path():
    p = os.path.join(output_root_path, "state", "openai_use.db")
    make_parent_exists(p)
    return p


def setup_file_logger(logger_name):
    log_file = os.path.join(output_root_path, "log", "{}.log".format(logger_name))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


import sqlite3


class TokenUsageDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_state_db_path()
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database and create table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT UNIQUE,
                    tokens_used INTEGER
                )
            ''')
            conn.commit()

    def add_usage(self, model, tokens_used):
        """Add to or update the token usage for a specific model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if an entry already exists for this model
            cursor.execute('SELECT tokens_used FROM token_usage WHERE model = ?', (model,))
            row = cursor.fetchone()

            if row:
                # If entry exists, update the token count by adding the new tokens_used
                current_tokens = row[0]
                new_total_tokens = current_tokens + tokens_used
                cursor.execute('''
                    UPDATE token_usage 
                    SET tokens_used = ? 
                    WHERE model = ?
                ''', (new_total_tokens, model))
            else:
                # If no entry exists, insert a new row
                cursor.execute('''
                    INSERT INTO token_usage (model, tokens_used) 
                    VALUES (?, ?)
                ''', (model, tokens_used))

            conn.commit()

    def get_all_usages(self):
        """Retrieve all records of token usage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM token_usage')
            rows = cursor.fetchall()
        return rows


class OpenAIChatClient:
    def __init__(self, model="gpt-4o", db_client=None):
        self.logger = setup_file_logger("openai")
        self.client = get_open_ai()
        self.model = model
        self.total_tokens_used = 0  # Track total tokens used
        self.last_request_tokens = 0  # Track tokens used in the last request
        self.db_client = TokenUsageDB()  # Accepts an instance of TokenUsageDB

    def request(self, message):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )

        # Track tokens used
        self.last_request_tokens = chat_completion.usage.total_tokens
        self.total_tokens_used += self.last_request_tokens

        # Log message with tokens
        msg = {
            "request": message,
            "response": chat_completion.choices[0].message.content,
            "tokens_used": self.last_request_tokens,
            "total_tokens_used": self.total_tokens_used
        }
        self.logger.info(json.dumps(msg))

        return chat_completion.choices[0].message.content

    def __del__(self):
        """Update the database with total token usage upon destruction."""
        print("{} tokens used.".format(self.total_tokens_used))
        if self.db_client:
            self.db_client.add_usage(self.model, self.total_tokens_used)


def get_open_ai_my():
    api_key_path = os.path.join(data_root_path, "openai_api_key.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()

    return OpenAI(api_key=key)


def get_open_ai(deployment=None):
    api_key_path = os.path.join(data_root_path, "openai_api_key_uva.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()
    azure_endpoint = "https://rtp2-shared.openai.azure.com"
    client = AzureOpenAI(api_key=key, api_version="2024-10-21",
                         azure_endpoint=azure_endpoint, azure_deployment=deployment)
    return client



class DummyChatClient:
    def __init__(self):
        self.total_tokens_used = 0  # Track total tokens used
        self.last_request_tokens = 0  # Track tokens used in the last request

    def request(self, message):
        char_per_token = 4

        self.last_request_tokens = int(len(message) // char_per_token)
        self.total_tokens_used += self.last_request_tokens
        return ""

    def __del__(self):
        print("{} tokens used.".format(self.total_tokens_used))