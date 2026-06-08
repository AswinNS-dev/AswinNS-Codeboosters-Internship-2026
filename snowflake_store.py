import os
import snowflake.connector

# ── hardcoded for this single-domain project ──────────────────────────────────
SNOWFLAKE_DATABASE = "HEALTHCARE_DB"
SNOWFLAKE_SCHEMA   = "PUBLIC"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"

class SnowflakeStore:
    def __init__(self):
        self.conn = snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            warehouse=SNOWFLAKE_WAREHOUSE,
        )
        self.cursor = self.conn.cursor()

    def init_table(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE}")
        self.cursor.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
        self.cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS HEALTHCARE_INTERACTIONS (
                QUERY_ID           VARCHAR(64)   PRIMARY KEY,
                USER_QUESTION      TEXT,
                AI_RESPONSE        TEXT,
                DOMAIN_CATEGORY    VARCHAR(100),
                DOMAIN_SUBCATEGORY VARCHAR(100),
                TIMESTAMP          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
                SESSION_ID         VARCHAR(64)
            )
        """)
        self.conn.commit()

    def save_interaction(self, query_id, user_question, ai_response,
                         domain_category, domain_subcategory, session_id):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO HEALTHCARE_INTERACTIONS "
            "(QUERY_ID, USER_QUESTION, AI_RESPONSE, DOMAIN_CATEGORY, DOMAIN_SUBCATEGORY, SESSION_ID) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(query_id), str(user_question), str(ai_response),
             str(domain_category), str(domain_subcategory), str(session_id))
        )
        self.conn.commit()

    def fetch_all_interactions(self):
        self.cursor.execute("""
            SELECT USER_QUESTION, AI_RESPONSE, DOMAIN_CATEGORY, DOMAIN_SUBCATEGORY
            FROM HEALTHCARE_INTERACTIONS
            ORDER BY TIMESTAMP ASC
        """)
        rows = self.cursor.fetchall()
        return [
            {
                "question": r[0],
                "response": r[1],
                "category": r[2],
                "subcategory": r[3],
            }
            for r in rows
        ]

    def get_stats(self):
        self.cursor.execute("SELECT COUNT(*) FROM HEALTHCARE_INTERACTIONS")
        total = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT DOMAIN_CATEGORY, COUNT(*) AS cnt
            FROM HEALTHCARE_INTERACTIONS
            GROUP BY DOMAIN_CATEGORY
            ORDER BY cnt DESC
        """)
        by_category = {row[0]: row[1] for row in self.cursor.fetchall()}

        return {"total": total, "by_category": by_category}

    def close(self):
        self.cursor.close()
        self.conn.close()