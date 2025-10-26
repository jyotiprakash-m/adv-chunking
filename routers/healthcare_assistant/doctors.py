from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import SQLModel, Field, Session, select, func

from utils.database import get_session

# Doctor model representing a medical doctor in the healthcare system
class Doctor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    specialist: str
    hospital: str
    contact_no: str
    email: str
    education: str

# Initialize the API router for healthcare assistant
router = APIRouter(prefix="/healthcare", tags=["Healthcare Assistant"])

# =========================================
# Healthcare Assistant Endpoints
# =========================================

# POST - Add doctor
@router.post("/doctors/", response_model=Doctor)
def add_doctor(doctor: Doctor, session: Session = Depends(get_session)):
    return create_doctor(doctor, session)

# GET - All doctors (optionally filter by specialist)
@router.get("/doctors/", response_model=list[Doctor])
def get_doctors(
    specialist: str | None = Query(None, description="Filter by specialization"),
    session: Session = Depends(get_session)
):
    if specialist:
        return find_doctors_by_specialization(specialist, session)
    return get_all_doctors(session)

# GET - Doctor by ID
@router.get("/doctors/{doctor_id}", response_model=Doctor)
def get_doctor(doctor_id: int, session: Session = Depends(get_session)):
    return find_doctor_by_id(doctor_id, session)

# POST - Add multiple doctors for testing
@router.post("/doctors/bulk/", response_model=list[Doctor])
def add_multiple_doctors(doctors: list[Doctor], session: Session = Depends(get_session)):
    return create_multiple_doctors(doctors, session)

# =========================================
# Services and Utilities
# =========================================

def create_doctor(doctor: Doctor, session: Session) -> Doctor:
    """Add a new doctor to the database."""
    session.add(doctor)
    session.commit()
    session.refresh(doctor)
    return doctor

def find_doctors_by_specialization(specialization: str, session: Session) -> list[Doctor]:
    """Find doctors by their specialization (case-insensitive, partial match)."""
    query = select(Doctor).where(func.lower(Doctor.specialist).like(f"%{specialization.lower()}%"))
    return list(session.exec(query).all())

def get_all_doctors(session: Session) -> list[Doctor]:
    """Get all doctors."""
    query = select(Doctor)
    return list(session.exec(query).all())

def find_doctor_by_id(doctor_id: int, session: Session) -> Doctor:
    """Find a doctor by their ID."""
    doctor = session.get(Doctor, doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor
# Create multiple doctors for testing
def create_multiple_doctors(doctors: list[Doctor], session: Session) -> list[Doctor]:
    """Add multiple doctors to the database."""
    session.add_all(doctors)
    session.commit()
    for doctor in doctors:
        session.refresh(doctor)
    return doctors