"""Database cleanup: dedup signals and delete old rows."""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def cleanup_database():
    """Remove duplicate signals and signals older than 90 days."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()

    # Remove duplicates (keep latest)
    cursor.execute(
        """
        DELETE FROM ai_company_signals
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_name, headline
                           ORDER BY created_at DESC
                       ) as rn
                FROM ai_company_signals
            ) t
            WHERE rn > 1
        )
    """
    )
    dedup_count = cursor.rowcount
    print(f"✅ Deleted {dedup_count} duplicate signals")

    # Delete signals older than 90 days
    cursor.execute(
        """
        DELETE FROM ai_company_signals
        WHERE created_at < NOW() - INTERVAL '90 days'
    """
    )
    old_count = cursor.rowcount
    print(f"✅ Deleted {old_count} signals older than 90 days")

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n✅ Cleanup complete: {dedup_count + old_count} total rows removed")


if __name__ == "__main__":
    cleanup_database()
