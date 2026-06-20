"""Repeat offender tracking using SQLite."""

import sqlite3
import os
import json
from datetime import datetime, timezone

__all__ = ['RepeatOffenderTracker']


class RepeatOffenderTracker:
    """
    Tracks repeat offenders by license plate number using SQLite.
    
    Maintains a local database of all violations keyed by plate number,
    allowing lookup of prior violation count for risk score calculation.
    """

    def __init__(self, db_path='evidence_store/offenders.db'):
        """
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_timestamp TEXT NOT NULL,
                evidence_id TEXT,
                camera_id TEXT,
                camera_location TEXT,
                confidence REAL,
                risk_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON offenders(plate_number)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plate_summary (
                plate_number TEXT PRIMARY KEY,
                total_violations INTEGER DEFAULT 0,
                first_violation TEXT,
                last_violation TEXT,
                violation_types TEXT DEFAULT '[]'
            )
        ''')

        conn.commit()
        conn.close()

    def record_violation(self, plate_number, violation_type, evidence_id=None,
                        camera_id=None, camera_location=None,
                        confidence=0.0, risk_score=0.0):
        """
        Record a new violation for a plate number.

        Args:
            plate_number: License plate text.
            violation_type: Type of violation.
            evidence_id: UUID of the evidence record.
            camera_id: Camera identifier.
            camera_location: Camera location name.
            confidence: Detection confidence.
            risk_score: Calculated risk score.

        Returns:
            int: Updated total violation count for this plate.
        """
        if not plate_number or plate_number.strip() == '':
            return 0

        plate_number = plate_number.strip().upper()
        timestamp = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert violation record
            cursor.execute('''
                INSERT INTO offenders 
                (plate_number, violation_type, violation_timestamp, evidence_id,
                 camera_id, camera_location, confidence, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (plate_number, violation_type, timestamp, evidence_id,
                  camera_id, camera_location, confidence, risk_score))

            # Update summary
            cursor.execute(
                'SELECT total_violations, violation_types FROM plate_summary WHERE plate_number = ?',
                (plate_number,)
            )
            row = cursor.fetchone()

            if row:
                total = row[0] + 1
                types = json.loads(row[1]) if row[1] else []
                if violation_type not in types:
                    types.append(violation_type)
                cursor.execute('''
                    UPDATE plate_summary 
                    SET total_violations = ?, last_violation = ?, violation_types = ?
                    WHERE plate_number = ?
                ''', (total, timestamp, json.dumps(types), plate_number))
            else:
                total = 1
                cursor.execute('''
                    INSERT INTO plate_summary 
                    (plate_number, total_violations, first_violation, last_violation, violation_types)
                    VALUES (?, ?, ?, ?, ?)
                ''', (plate_number, 1, timestamp, timestamp, json.dumps([violation_type])))

            conn.commit()
            return total

        except Exception as e:
            conn.rollback()
            print(f"[RepeatOffender] Error recording violation: {e}")
            return 0
        finally:
            conn.close()

    def get_violation_count(self, plate_number):
        """
        Get the total number of prior violations for a plate number.

        Args:
            plate_number: License plate text.

        Returns:
            int: Total violation count (0 if not found).
        """
        if not plate_number or plate_number.strip() == '':
            return 0

        plate_number = plate_number.strip().upper()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT total_violations FROM plate_summary WHERE plate_number = ?',
            (plate_number,)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else 0

    def get_offender_history(self, plate_number):
        """
        Get full violation history for a plate number.

        Returns:
            list[dict]: List of violation records.
        """
        if not plate_number or plate_number.strip() == '':
            return []

        plate_number = plate_number.strip().upper()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM offenders WHERE plate_number = ? ORDER BY violation_timestamp DESC',
            (plate_number,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_top_offenders(self, limit=10):
        """
        Get the top repeat offenders by violation count.

        Returns:
            list[dict]: List of {plate_number, total_violations, ...}.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM plate_summary 
            ORDER BY total_violations DESC 
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self):
        """Get overall offender statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM plate_summary')
        unique_plates = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM offenders')
        total_violations = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM plate_summary WHERE total_violations > 1')
        repeat_offenders = cursor.fetchone()[0]

        conn.close()

        return {
            'unique_plates': unique_plates,
            'total_violations': total_violations,
            'repeat_offenders': repeat_offenders
        }
