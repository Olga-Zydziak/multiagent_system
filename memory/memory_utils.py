from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
import vertexai
import langchain
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from tools.utils import intelligent_truncate
from config import MAIN_AGENT,LOCATION,PROJECT_ID,CRITIC_MODEL
from .memory_models import *

#--narzedzie do przetwarzania info dla pamieci dlugotrwalej, llm agent uzywa llm!!
def distill_memory_content(failing_code: str, error_traceback: str, debugger_analysis: str, corrected_code: str) -> dict:
    """Używa LLM do 'przedestylowania' surowych danych o błędzie i jego naprawie do zwięzłego, ustrukturyzowanego formatu."""
    print("INFO: Uruchamiam proces destylacji wspomnienia (wersja ekspercka)...")
    
    prompt_template = f"""
    Persona: Jesteś starszym inżynierem oprogramowania, który pisze zwięzłe post-mortemy do wewnętrznej bazy wiedzy. Twoim celem jest stworzenie notatki, która będzie maksymalnie użyteczna dla innych agentów w przyszłości.
    Przeanalizuj poniższy kontekst i wyciągnij z niego kluczowe, gotowe do użycia wnioski.
    Kontekst:
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


    
    
    
    
# --- FUNKCJA 1: Destylacja wspomnienia o UDANEJ NAPRAWIE ---
def distill_fix_memory(initial_error: str, fix_attempts: List[Dict], successful_code: str) -> Dict[str, Any]:
    """
    Używa LLM, aby podsumować całą, kompletną sesję naprawczą w jedno zwięzłe wspomnienie.
    Inteligentnie skraca duże fragmenty danych PRZED wysłaniem ich do LLM.
    """
    print("  [PAMIĘĆ] Uruchamiam destylację całej sesji naprawczej...")

    truncated_error = intelligent_truncate(initial_error, 1500)
    truncated_code = intelligent_truncate(successful_code, 2000)

    consolidated_analysis = "\n".join(
        [f"Próba {i+1}: {intelligent_truncate(attempt.get('debugger_analysis', 'Brak analizy.'), 500)}" for i, attempt in enumerate(fix_attempts)]
    )

    prompt_template = f"""
Persona: Jesteś starszym inżynierem AI, który pisze ekstremalnie zwięzłe post-mortemy. Twoim celem jest stworzenie notatki, która będzie maksymalnie użyteczna dla innych w przyszłości. Priorytetem jest gęstość informacji.

Przeanalizuj poniższą sesję naprawy błędu i wyciągnij z niej kluczowe wnioski.

<PIERWOTNY_BŁĄD>
{truncated_error}
</PIERWOTNY_BŁĄD>

<HISTORIA_ANALIZ_Z_NIEUDANYCH_PRÓB>
{consolidated_analysis}
</HISTORIA_ANALIZ_Z_NIEUDANYCH_PRÓB>

<KOD_KTORY_ZADZIAŁAŁ>
{truncated_code}
</KOD_KTORY_ZADZIAŁAŁ>

Zadanie: Wygeneruj obiekt JSON. Każde pole tekstowe musi być pojedynczym, klarownym zdaniem. Unikaj ogólników.
"""
    try:
        llm = ChatVertexAI(
            model_name=MAIN_AGENT,
            project_id=PROJECT_ID,
            location=LOCATION,
            max_output_tokens=1024
        )
        structured_llm = llm.with_structured_output(DistilledMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("  [PAMIĘĆ] ✅ Pomyślnie przedestylowano wspomnienie o naprawie.")
        return distilled_object.dict()
    except Exception as e:
        print(f"  [PAMIĘĆ] ⚠️ OSTRZEŻENIE: Destylacja sesji nie powiodła się: {e}.")
        return {
            "problem_summary": "Distillation failed. The initial error was likely related to code execution.",
            "solution_summary": "A fix was applied, but could not be summarized.",
            "key_takeaway": "N/A - distillation process failed.",
            "tags": ["error", "distillation-failure"],
            "reusable_code_snippet": None
        }

    
    
    
def distill_successful_workflow(plan: str, final_code: str, langgraph_log: str) -> dict:
    """Używa LLM do podsumowania całego udanego procesu w jedną, bogatą notatkę."""
    print("  [PAMIĘĆ] Uruchamiam destylację całego udanego procesu...")

    truncated_plan = intelligent_truncate(plan, 2000)
    truncated_code = intelligent_truncate(final_code, 2000)
    truncated_log = intelligent_truncate(langgraph_log, 3000)

    prompt_template = f"""
Persona: Jesteś starszym inżynierem AI, który dokumentuje udane projekty w całości.
Kontekst: Przeanalizuj poniższe artefakty z pomyślnie zakończonego procesu przetwarzania danych. Twoim zadaniem jest wyciągnięcie esencji sukcesu zarówno z fazy planowania, jak i wykonania.

<FINALNY_PLAN>
{truncated_plan}
</FINALNY_PLAN>

<FINALNY_WYKONANY_KOD>
{truncated_code}
</FINALNY_WYKONANY_KOD>

<LOG_WYKONANIA>
{truncated_log}
</LOG_WYKONANIA>

Zadanie: Wygeneruj obiekt JSON zgodny ze strukturą `DistilledWorkflowMemory`. Rozróżnij wniosek z planowania od obserwacji z wykonania.
"""
    try:
        llm = ChatVertexAI(model_name=MAIN_AGENT, project_id=PROJECT_ID, location=LOCATION)
        structured_llm = llm.with_structured_output(DistilledWorkflowMemory)
        distilled_object = structured_llm.invoke(prompt_template)
        print("  [PAMIĘĆ] ✅ Pomyślnie przedestylowano wspomnienie o całym procesie.")
        return distilled_object.dict()
    except Exception as e:
        print(f"  [PAMIĘĆ] ⚠️ OSTRZEŻENIE: Destylacja procesu nie powiodła się: {e}.")
        return {"workflow_summary": "N/A - distillation failed"}

    
def generate_meta_insight(audit_report: str) -> Optional[dict]:
    """Używa LLM do wyciągnięcia z raportu audytora jednego, kluczowego wniosku."""
    print("  [AUDYT] Uruchamiam proces generowania wniosku META...")
    prompt = f"""
Przeanalizuj poniższy raport audytora. Twoim zadaniem jest znalezienie JEDNEJ, najważniejszej i najbardziej konkretnej rekomendacji dotyczącej ulepszenia systemu.
Jeśli znajdziesz taką rekomendację, przekształć ją w obiekt JSON zgodny ze strukturą MetaInsightMemory.
Jeśli raport jest ogólnikowy i nie zawiera konkretnych propozycji, zwróć null.
[RAPORT AUDYTORA]:\n{audit_report}
"""
    try:
        llm = ChatAnthropic(model_name=CRITIC_MODEL, temperature=0.2)
        structured_llm = llm.with_structured_output(MetaInsightMemory)
        insight_object = structured_llm.invoke(prompt)
        print("  [AUDYT] ✅ Pomyślnie wygenerowano wniosek META.")
        return insight_object.dict()
    except Exception as e:
        print(f"  [AUDYT] ⚠️ OSTRZEŻENIE: Nie udało się wygenerować wniosku META z raportu audytora. Błąd: {e}")
        return None    
    

