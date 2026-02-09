"""Import Synthea CSV data into SQLite for the Lens prototype."""

import csv
import os
import sys
from datetime import datetime, date
from collections import Counter
from pathlib import Path

# Ensure we can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import engine, SessionLocal, Base
from models import Patient, VitalSign, LabResult, ClinicalNote, MedicationLog

DATA_DIR = Path(__file__).parent / "synthea_output"

# LOINC codes → vital_type mapping
VITAL_CODES = {
    "8867-4": ("heart_rate", "/min"),
    "8480-6": ("systolic_bp", "mm[Hg]"),
    "8462-4": ("diastolic_bp", "mm[Hg]"),
    "29463-7": ("weight", "kg"),
    "8310-5": ("temperature", "Cel"),
    "9279-1": ("respiratory_rate", "/min"),
    "2708-6": ("o2_saturation", "%"),
}

# LOINC codes → lab test name mapping
LAB_CODES = {
    "2160-0": "creatinine",
    "38483-4": "creatinine_blood",
    "2951-2": "sodium",
    "2947-0": "sodium_blood",
    "2823-3": "potassium",
    "6298-4": "potassium_blood",
    "89579-7": "troponin",
    "10230-1": "ejection_fraction",
    "2339-0": "glucose",
    "2345-7": "glucose_serum",
    "4548-4": "hemoglobin_a1c",
    "2093-3": "cholesterol_total",
    "2085-9": "cholesterol_hdl",
    "18262-6": "cholesterol_ldl",
    "2571-8": "triglycerides",
    "3094-0": "bun",
    "6299-2": "bun_blood",
    "718-7": "hemoglobin",
    "4544-3": "hematocrit",
    "33914-3": "gfr",
    "2885-2": "total_protein",
    "1742-6": "alt",
    "1920-8": "ast",
    "1751-7": "albumin",
    "1975-2": "bilirubin",
    "777-3": "platelets",
    "6690-2": "wbc",
    "789-8": "rbc",
}


def parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def find_best_patient() -> str:
    """Find the heart-disease patient with the richest data."""
    obs_counts: Counter = Counter()
    cond_counts: Counter = Counter()
    med_counts: Counter = Counter()

    with open(DATA_DIR / "observations.csv") as f:
        for row in csv.DictReader(f):
            obs_counts[row["PATIENT"]] += 1

    with open(DATA_DIR / "conditions.csv") as f:
        for row in csv.DictReader(f):
            cond_counts[row["PATIENT"]] += 1

    with open(DATA_DIR / "medications.csv") as f:
        for row in csv.DictReader(f):
            med_counts[row["PATIENT"]] += 1

    heart_patients = set()
    with open(DATA_DIR / "conditions.csv") as f:
        for row in csv.DictReader(f):
            desc = row["DESCRIPTION"].lower()
            if any(w in desc for w in ("heart", "cardiac", "coronary", "hypertens")):
                heart_patients.add(row["PATIENT"])

    scores = {}
    for pid in obs_counts:
        scores[pid] = obs_counts[pid] + cond_counts.get(pid, 0) * 10 + med_counts.get(pid, 0) * 5

    # Prefer heart patients; fall back to richest overall
    if heart_patients:
        heart_scores = {p: scores.get(p, 0) for p in heart_patients}
        return max(heart_scores, key=heart_scores.get)
    return max(scores, key=scores.get)


def import_patient(db, patient_id: str):
    """Import a single patient and all related data."""
    # ── Patient ──
    with open(DATA_DIR / "patients.csv") as f:
        for row in csv.DictReader(f):
            if row["Id"] == patient_id:
                # Find primary diagnosis — prefer cardiac/serious conditions
                primary_dx = None
                cardiac_keywords = ("heart", "cardiac", "coronary", "myocardial", "hypertens", "infarction")
                all_active = []
                with open(DATA_DIR / "conditions.csv") as cf:
                    for crow in csv.DictReader(cf):
                        if crow["PATIENT"] == patient_id and not crow["STOP"]:
                            all_active.append(crow["DESCRIPTION"])
                # Pick cardiac condition first, then any disorder
                for desc in all_active:
                    if any(k in desc.lower() for k in cardiac_keywords):
                        primary_dx = desc
                        break
                if not primary_dx:
                    for desc in all_active:
                        if "disorder" in desc.lower():
                            primary_dx = desc
                            break
                if not primary_dx and all_active:
                    primary_dx = all_active[0]

                patient = Patient(
                    id=patient_id,
                    first_name=row["FIRST"],
                    last_name=row["LAST"],
                    birth_date=parse_date(row["BIRTHDATE"]),
                    gender=row["GENDER"],
                    race=row.get("RACE", ""),
                    city=row.get("CITY", ""),
                    state=row.get("STATE", ""),
                    primary_diagnosis=primary_dx,
                )
                db.add(patient)
                print(f"  Patient: {row['FIRST']} {row['LAST']} ({row['GENDER']}, born {row['BIRTHDATE']})")
                print(f"  Primary diagnosis: {primary_dx}")
                break

    # ── Observations → Vitals + Labs ──
    vitals_count = 0
    labs_count = 0
    with open(DATA_DIR / "observations.csv") as f:
        for row in csv.DictReader(f):
            if row["PATIENT"] != patient_id:
                continue
            code = row["CODE"]
            value_str = row["VALUE"]
            recorded_at = parse_dt(row["DATE"])

            # Try to parse as float
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue

            if code in VITAL_CODES:
                vtype, unit = VITAL_CODES[code]
                db.add(VitalSign(
                    patient_id=patient_id,
                    recorded_at=recorded_at,
                    vital_type=vtype,
                    value=value,
                    unit=unit,
                ))
                vitals_count += 1
            elif code in LAB_CODES:
                db.add(LabResult(
                    patient_id=patient_id,
                    recorded_at=recorded_at,
                    test_name=LAB_CODES[code],
                    value=value,
                    unit=row.get("UNITS", ""),
                    loinc_code=code,
                ))
                labs_count += 1

    print(f"  Vitals: {vitals_count} records")
    print(f"  Labs: {labs_count} records")

    # ── Encounters → Clinical Notes ──
    notes_count = 0
    with open(DATA_DIR / "encounters.csv") as f:
        for row in csv.DictReader(f):
            if row["PATIENT"] != patient_id:
                continue
            db.add(ClinicalNote(
                patient_id=patient_id,
                encounter_date=parse_dt(row["START"]),
                encounter_class=row.get("ENCOUNTERCLASS", ""),
                description=row.get("DESCRIPTION", ""),
                reason=row.get("REASONDESCRIPTION", ""),
            ))
            notes_count += 1
    print(f"  Clinical notes: {notes_count} records")

    # ── Medications ──
    meds_count = 0
    with open(DATA_DIR / "medications.csv") as f:
        for row in csv.DictReader(f):
            if row["PATIENT"] != patient_id:
                continue
            db.add(MedicationLog(
                patient_id=patient_id,
                start_date=parse_dt(row["START"]),
                stop_date=parse_dt(row["STOP"]) if row.get("STOP") else None,
                description=row.get("DESCRIPTION", ""),
                reason=row.get("REASONDESCRIPTION", ""),
            ))
            meds_count += 1
    print(f"  Medications: {meds_count} records")

    db.commit()


def main():
    print("=" * 60)
    print("Lens — Synthea Data Import")
    print("=" * 60)

    # Check data files exist
    required = ["patients.csv", "conditions.csv", "observations.csv", "medications.csv", "encounters.csv"]
    for fname in required:
        if not (DATA_DIR / fname).exists():
            print(f"ERROR: Missing {DATA_DIR / fname}")
            print("Download Synthea sample data first.")
            sys.exit(1)

    # Create tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    # Find best patient
    patient_id = find_best_patient()
    print(f"\nSelected patient: {patient_id}")

    # Import
    db = SessionLocal()
    try:
        import_patient(db, patient_id)
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("Import complete! Run the server with:")
    print("  ANTHROPIC_API_KEY=sk-... uvicorn main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
