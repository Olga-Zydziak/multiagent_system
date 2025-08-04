from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
import vertexai
import langchain
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from tools.utils import intelligent_truncate
from config import MAIN_AGENT,LOCATION,PROJECT_ID,CRITIC_MODEL
#--narzedzie do przetwarzania info dla pamieci dlugotrwalej, llm agent uzywa llm!!
def distill_memory_content(failing_code: str, error_traceback: str, debugger_analysis: str, corrected_code: str) -> dict:
    """Używa LLM do 'przedestylowania' surowych danych o błędzie i jego naprawie do zwięzłego, ustrukturyzowanego formatu."""
    print("INFO: Uruchamiam proces destylacji wspomnienia (wersja ekspercka)...")
    
    prompt_template = f"""
    Persona: Jesteś starszym inżynierem oprogramowania, który pisze zwięzłe post-mortemy do wewnętrznej bazy wiedzy. Twoim celem jest stworzenie notatki, która będzie maksymalnie użyteczna dla innych agentów w przyszłości.
    Przeanalizuj poniższy kontekst i wyciągnij z niego kluczowe, gotowe do użycia wnioski.
    Kontekst:888888888888888
    [WADLIWY KOD]: {failing_code}
    [PEŁNY BŁĄD]: {error_traceback}
    [ANALIZA PROBLEMU]: {debugger_analysis}
    [POPRAWIONY KOD]: {corrected_code}
    Zadanie: Na podstawie powyższego kontekstu, wygeneruj obiekt, który będzie pasował do zdefiniowanej struktury.
    """
    
    try:
        llm = ChatVertexAI(model_name=MAIN_AGENT, project_id=PROJECT_ID, location=LOCATION)
        structured_llm = llm.with_structured_output(DistilledMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("INFO: Pomyślnie przedestylowano wspomnienie (wersja ekspercka).")
        return distilled_object.dict()
    except Exception as e:
        print(f"OSTRZEŻENIE: Destylacja (ekspercka) nie powiodła się: {e}. Zapisuję surowe dane.")
        return {
            "problem_summary": debugger_analysis,
            "key_takeaway": "N/A - distillation failed",
            "raw_error": intelligent_truncate(error_traceback, 500)
        }
#pamiec dlugotrwala-zapis w meta agent, sukces 
def distill_success_memory(final_plan: str) -> dict:
    """Używa LLM do podsumowania udanego planu w zwięzłą notatkę."""
    print("INFO: Uruchamiam proces destylacji wspomnienia o sukcesie...")
    prompt_template = f"""
    Persona: Jesteś starszym inżynierem AI, który dokumentuje udane strategie.
    Kontekst: Przeanalizuj poniższy plan, który zakończył się sukcesem i stwórz zwięzłe podsumowanie w formacie JSON.
    [FINALNY PLAN]: {final_plan}
    """
    try:
        llm = ChatVertexAI(model_name=MAIN_AGENT, project_id=PROJECT_ID, location=LOCATION)
        # Używamy nowego, lżejszego modelu DistilledSuccessMemory
        structured_llm = llm.with_structured_output(DistilledSuccessMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("INFO: Pomyślnie przedestylowano wspomnienie o sukcesie.")
        return distilled_object.dict()
    except Exception as e:
        print(f"OSTRZEŻENIE: Destylacja sukcesu nie powiodła się: {e}.")
        return {"plan_summary": "N/A - distillation failed"}
    
    
def distill_memory_content(debugger_analysis: str, failing_code: str, corrected_code: str) -> dict:
    """Używa LLM do 'przedestylowania' analizy debuggera i zmian w kodzie do zwięzłego formatu."""
    print("INFO: Uruchamiam proces destylacji wspomnienia o naprawie...")
    
    # Zamiast pełnego błędu, używamy zwięzłej analizy od debuggera!
    prompt_template = f"""
    Persona: Jesteś starszym inżynierem oprogramowania, który pisze zwięzłe post-mortemy.
    Przeanalizuj poniższy kontekst dotyczący naprawy błędu i wyciągnij z niego kluczowe, gotowe do użycia wnioski.
    Kontekst:
    [ANALIZA PROBLEMU WG DEBUGGERA]: {debugger_analysis}
    [WADLIWY FRAGMENT KODU]: {intelligent_truncate(failing_code, 500)}
    [POPRAWIONY KOD]: {intelligent_truncate(corrected_code, 500)}
    Zadanie: Na podstawie powyższego kontekstu, wygeneruj obiekt JSON zgodny ze strukturą DistilledMemory.
    """
    
    try:
        llm = ChatVertexAI(model_name=MAIN_AGENT, project_id=PROJECT_ID, location=LOCATION)
        structured_llm = llm.with_structured_output(DistilledMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("INFO: Pomyślnie przedestylowano wspomnienie o naprawie.")
        return distilled_object.dict()
    except Exception as e:
        print(f"OSTRZEŻENIE: Destylacja (naprawa) nie powiodła się: {e}.")
        return {"key_takeaway": "N/A - distillation failed"}
    
    
    
    

    
def distill_full_fix_session(initial_error: str, fix_attempts: List[Dict], successful_code: str) -> Dict[str, Any]:
    """Używa LLM, aby podsumować całą sesję naprawczą w jedno zwięzłe wspomnienie."""
    print("  [INFO] Uruchamiam destylację całej sesji naprawczej...")

    # Tworzymy skonsolidowaną historię analiz debuggera
    consolidated_analysis = "\n".join(
        [f"Próba {i+1}: {attempt.get('debugger_analysis', 'Brak analizy.')}" for i, attempt in enumerate(fix_attempts)]
    )

    prompt_template = f"""
Persona: Jesteś starszym inżynierem, który pisze ekstremalnie zwięzłe post-mortemy. Priorytetem jest gęstość informacji przy minimalnej liczbie słów.

Przeanalizuj całą sesję naprawy błędu i wyciągnij z niej kluczowe wnioski.

[PIERWOTNY BŁĄD]:
{initial_error}

[HISTORIA ANALIZ Z NIEUDANYCH PRÓB NAPRAWY]:
{consolidated_analysis}

[KOD, KTÓRY OSTATECZNIE ZADZIAŁAŁ]:
{successful_code}

Zadanie: Wygeneruj obiekt JSON. Każde pole tekstowe musi być pojedynczym, klarownym zdaniem. Całość nie może przekroczyć 150 słów.
"""
    try:
        llm = ChatVertexAI(
            model_name=MAIN_AGENT, 
            project_id=PROJECT_ID, 
            location=LOCATION,
            max_output_tokens=512
        )
        structured_llm = llm.with_structured_output(DistilledMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("  [INFO] Pomyślnie przedestylowano wspomnienie o naprawie.")
        return distilled_object.dict()
    except Exception as e:
        print(f"  [OSTRZEŻENIE] Destylacja sesji nie powiodła się: {e}.")
        return {"key_takeaway": "N/A - distillation failed"}



    
def generate_meta_insight(audit_report: str) -> Optional[dict]:
    """Używa LLM do wyciągnięcia z raportu audytora jednego, kluczowego wniosku."""
    print("INFO: Uruchamiam proces generowania wniosku META...")
    prompt = f"""
    Przeanalizuj poniższy raport audytora. Twoim zadaniem jest znalezienie JEDNEJ, najważniejszej i najbardziej konkretnej rekomendacji dotyczącej ulepszenia systemu.
    Jeśli znajdziesz taką rekomendację, przekształć ją w obiekt JSON zgodny ze strukturą MetaInsightMemory. Jeśli raport jest ogólnikowy i nie zawiera konkretnych propozycji, zwróć null.
    [RAPORT AUDYTORA]:\n{audit_report}
    """
    try:
        llm = ChatVertexAI(model_name=CRITIC_MODEL, project_id=PROJECT_ID, location=LOCATION)
        structured_llm = llm.with_structured_output(MetaInsightMemory)
        insight_object = structured_llm.invoke(prompt)
        print("INFO: Pomyślnie wygenerowano wniosek META.")
        return insight_object.dict()
    except Exception:
        print("OSTRZEŻENIE: Nie udało się wygenerować wniosku META z raportu audytora.")
        return None