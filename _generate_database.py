import sqlite3

def generate_database(DB_PATH, STIM_LIB_PATH):
    from SiNAPSE.core import Database

    db=Database(DB_PATH, STIM_LIB_PATH)
    return db

def ensure_column_exists(conn, table, column, col_type="REAL"):
    cur = conn.cursor()

    # get existing columns
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]

    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()

def insert(DB_PATH, unit_id, session_id, value, column="value"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    table = "neurons"

    # ensure column exists
    ensure_column_exists(conn, table, column, col_type="REAL")

    # 1. CHECK EXISTENCE FIRST (STRICT MODE)
    cur.execute(f"""
        SELECT COUNT(*)
        FROM {table}
        WHERE unit_id = ? AND session_id = ?
    """, (unit_id, session_id))

    exists = cur.fetchone()[0]

    if exists == 0:
        conn.close()
        raise ValueError(
            f"[DB ERROR] No existing row for unit_id={unit_id}, session_id={session_id}"
        )

    if exists > 1:
        conn.close()
        raise ValueError(
            f"[DB ERROR] Duplicate rows found for unit_id={unit_id}, session_id={session_id}"
        )

    # 2. SAFE UPDATE ONLY
    cur.execute(f"""
        UPDATE {table}
        SET {column} = ?
        WHERE unit_id = ? AND session_id = ?
    """, (value, unit_id, session_id))

    conn.commit()
    conn.close()

    # print(f"Updated {column}={value} @ unit_id={unit_id}, session_id={session_id}")

def session_is_complete(DB_PATH, session_id, column="value"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    table = "neurons"

    ensure_column_exists(conn, table, column, col_type="REAL")

    # total rows for this session
    cur.execute(f"""
        SELECT COUNT(*)
        FROM {table}
        WHERE session_id = ?
    """, (session_id,))
    total = cur.fetchone()[0]

    # rows where value is missing
    cur.execute(f"""
        SELECT COUNT(*)
        FROM {table}
        WHERE session_id = ?
        AND manual_isi_0_7 < 1
        AND {column} IS NULL
    """, (session_id,))
    missing = cur.fetchone()[0]

    conn.close()

    # True only if nothing is missing
    # print(total > 0 and missing == 0)
    return total > 0 and missing == 0