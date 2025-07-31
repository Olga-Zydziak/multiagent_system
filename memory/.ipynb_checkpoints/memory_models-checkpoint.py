from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import uuid

class MemoryType(str, Enum):
    SUCCESSFUL_PLAN = "SUCCESSFUL_PLAN"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    SUCCESSFUL_FIX = "SUCCESSFUL_FIX"
    META_INSIGHT = "META_INSIGHT"

    

class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str # Dodajemy ID bieżącego uruchomienia
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: MemoryType
    dataset_signature: str
    source_node: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DistilledMemory(BaseModel):
    """Ustrukturyzowany, 'ekspercki' format dla przedestylowanego wspomnienia."""
    problem_summary: str = Field(description="Opis problemu w jednym, zwięzłym zdaniu.")
    solution_summary: str = Field(description="Opis rozwiązania w jednym, zwięzłym zdaniu.")
    applicability_context: str = Field(description="Opis w jednym zdaniu, w jakim kontekście (np. typ danych, operacja) ta lekcja jest najbardziej użyteczna.")
    key_takeaway: str = Field(description="Uniwersalna 'złota myśl' lub reguła na przyszłość, aby uniknąć podobnych błędów.")
    reusable_code_snippet: Optional[str] = Field(description="Generyczny fragment kodu w Pythonie (do 10 linijek), który implementuje 'key_takeaway'. Jeśli nie ma zastosowania, zwróć null.")
    tags: List[str] = Field(description="Lista 3-5 słów kluczowych (tagów) opisujących ten problem.")
    
    
 #--czysty zapis o sukcesie   
class DistilledSuccessMemory(BaseModel):
    """Ustrukturyzowany format dla wspomnienia o udanym przebiegu."""
    plan_summary: str = Field(description="Podsumowanie celu i kluczowych kroków zrealizowanego planu w jednym zdaniu.")
    key_insight: str = Field(description="Najważniejszy wniosek lub 'trick', który przyczynił się do sukcesu tego planu.")
    full_plan_text: str = Field(description="Pełny, szczegółowy, numerowany tekst udanego planu.")
    tags: List[str] = Field(description="Lista 3-5 słów kluczowych (tagów) opisujących ten plan.")
    
class MetaInsightMemory(BaseModel):
    """Ustrukturyzowany format dla WNIOSKU NA POZIOMIE SYSTEMU."""
    observation: str = Field(description="Zwięzłe opisanie zaobserwowanego zjawiska, np. 'Agent generujący raporty często popełniał błędy w wizualizacji danych szeregów czasowych'.")
    recommendation: str = Field(description="Konkretna, pojedyncza propozycja zmiany w prompcie lub logice systemu, np. 'Do promptu agenta raportującego należy dodać konkretny przykład użycia      `plt.xticks(rotation=45)`'.")
    target_agent_or_node: str = Field(description="Nazwa agenta lub węzła, którego dotyczy rekomendacja, np. 'reporting_agent_node'.")
    tags: List[str] = Field(description="Lista 3-5 słów kluczowych, np. ['prompt-engineering', 'reporting', 'visualization'].")
    
    
    