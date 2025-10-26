from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import SQLModel, Field, Session, select, func

from utils.database import get_session

# Patient model representing a patient in the healthcare system
class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    registration_no: str
    name: str
    address: str
    contact_no: str
    email: str
    registration_date: str

# Initialize the API router for healthcare assistant
router = APIRouter(prefix="/healthcare", tags=["Healthcare Assistant"])

# =========================================
# Healthcare Assistant Endpoints
# =========================================

# POST - Add patient
@router.post("/patients/", response_model=Patient)
def add_patient(patient: Patient, session: Session = Depends(get_session)):
    return create_patient(patient, session)

# GET - All patients (optionally filter by name)
@router.get("/patients/", response_model=list[Patient])
def get_patients(
    name: str | None = Query(None, description="Filter by name"),
    session: Session = Depends(get_session)
):
    if name:
        return find_patients_by_name(name, session)
    return get_all_patients(session)

# GET - Patient by ID
@router.get("/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: int, session: Session = Depends(get_session)):
    return find_patient_by_id(patient_id, session)

# POST - Add multiple patients for testing
@router.post("/patients/bulk/", response_model=list[Patient])
def add_multiple_patients(patients: list[Patient], session: Session = Depends(get_session)):
    return create_multiple_patients(patients, session)

# =========================================
# Services and Utilities
# =========================================

def create_patient(patient: Patient, session: Session) -> Patient:
    """Add a new patient to the database."""
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient

def find_patients_by_name(name: str, session: Session) -> list[Patient]:
    """Find patients by their name (case-insensitive, partial match)."""
    query = select(Patient).where(func.lower(Patient.name).like(f"%{name.lower()}%"))
    return list(session.exec(query).all())

def get_all_patients(session: Session) -> list[Patient]:
    """Get all patients."""
    query = select(Patient)
    return list(session.exec(query).all())

def find_patient_by_id(patient_id: int, session: Session) -> Patient:
    """Find a patient by their ID."""
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

def create_multiple_patients(patients: list[Patient], session: Session) -> list[Patient]:
    """Add multiple patients to the database."""
    session.add_all(patients)
    session.commit()
    for patient in patients:
        session.refresh(patient)
    return patients