import os
import io
import sys
import subprocess
import tempfile
import traceback
import uuid
import json
import re
import matplotlib.pyplot as plt
from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
import pandas as pd
import langchain
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from .state import AgentWorkflowState
from prompts import LangchainAgentsPrompts
from tools.utils import *
from tools.langchain_tools import *
from prompts import ArchitecturalRule, ArchitecturalRulesManager,ARCHITECTURAL_RULES
from prompts_beta import PromptFactory
from config import MAX_CORRECTION_ATTEMPTS, PROJECT_ID,LOCATION
from memory.memory_utils import *
from memory.memory_models import *
# --- Definicje węzłów LangGraph ---


CODE_ARTIFACT_MAP = {
    # Węzły odpowiedzialne za główny kod przetwarzania danych
    "code_generator": "generated_code",
    "architectural_validator": "generated_code",
    "data_code_executor": "generated_code",
    
    # Węzły odpowiedzialne za kod do generowania wykresów
    "plot_generator_node": "plot_generation_code",
    "report_composer_node": "plot_generation_code", # Błąd w composerze jest najczęściej błędem w kodzie do wykresów
    
    # Węzeł odpowiedzialny za podsumowanie HTML
    "summary_analyst_node": "summary_html",
}





def schema_reader_node(state: AgentWorkflowState):
    print("--- WĘZEŁ: ANALIZATOR SCHEMATU DANYCH ---")
    print(f"DEBUG: Próbuję odczytać plik ze ścieżki: {state.get('input_path')}")
    try:
        df_header = pd.read_csv(state['input_path'], nrows=0)
        
        #pamięć długotrwała, tworzenie sygnatury
        memory_client = state['memory_client']
        dataset_signature = memory_client.create_dataset_signature(df_header)
        print(f"INFO: Wygenerowano sygnaturę danych: {dataset_signature}")
        #--koniec--
        
        return {"available_columns": df_header.columns.tolist(),"dataset_signature": dataset_signature}
    except Exception as e:
        return {"error_message": f"Błąd odczytu pliku: {e}", "failing_node": "schema_reader"}

def code_generator_node(state: AgentWorkflowState):
    """Generuje główny skrypt przetwarzający dane z użyciem structured output."""
    print("---  WĘZEŁ: GENERATOR KODU ---")
    try:
        CODE_MODEL = state['config']['CODE_MODEL']
        
        # ZMIANA: Znacząco zwiększamy max_tokens, aby model miał miejsce na wygenerowanie pełnego kodu.
        llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=4096)
        
        # Powiązanie LLM ze schematem Pydantic, aby wymusić poprawny format wyjściowy.
        structured_llm = llm.with_structured_output(GeneratedCode)
        
        prompt = PromptFactory.for_code_generator(
            plan=state['plan'], 
            available_columns=state['available_columns']
        )
        
        # Wywołanie zwraca obiekt Pydantic, a nie surowy string.
        response_object = structured_llm.invoke(prompt)
        
        # Używamy poprawnej i ujednoliconej nazwy pola: 'code'.
        code = response_object.code
        
        print("\nAgent-Analityk wygenerował następujący kod:")
        print("--------------------------------------------------")
        print(code)
        print("--------------------------------------------------")
        
        return {"generated_code": code}

    except Exception as e:
        # Dodajemy obsługę błędu, aby dać więcej kontekstu, jeśli coś pójdzie nie tak.
        print(f"BŁĄD KRYTYCZNY w code_generator_node podczas wywołania LLM: {e}")
        # Zwracamy błąd do stanu, aby graf mógł na niego zareagować.
        return {
            "error_message": f"Błąd podczas generowania kodu: {e}", 
            "failing_node": "code_generator",
            "error_context_code": state.get('plan', 'Brak planu w stanie do analizy.')
        }




def architectural_validator_node(state: AgentWorkflowState):
    print("--- 🛡️ WĘZEŁ: STRAŻNIK ARCHITEKTURY 🛡️ ---")
    code_to_check = state.get('generated_code', '')
    if not code_to_check:
        error_message = "Brak kodu do walidacji."
        print(f"  [WERDYKT] ❌ {error_message}")
        return {"error_message": error_message, "failing_node": "architectural_validator", "error_context_code": "", "correction_attempts": state.get('correction_attempts', 0) + 1}

    errors = [rule["error_message"] for rule in ARCHITECTURAL_RULES if rule["check"](code_to_check)]
    
    if errors:
        error_message = "Błąd Walidacji Architektonicznej: " + " ".join(errors)
        # <<< WAŻNY PRINT >>>
        print(f"  [WERDYKT] ❌ Kod łamie zasady architektury: {' '.join(errors)}")
        
        pending_session = {
            "initial_error": error_message,  # Używamy błędu walidacji jako błędu początkowego
            "initial_code": code_to_check,
            "fix_attempts": []
        }
        
        return {"error_message": error_message, "failing_node": "architectural_validator", "error_context_code": code_to_check, "correction_attempts": state.get('correction_attempts', 0) + 1}
    else:
        # <<< WAŻNY PRINT >>>
        print("  [WERDYKT] Kod jest zgodny z architekturą systemu.")
        return {"error_message": None, "pending_fix_session": None}

    
def data_code_executor_node(state: AgentWorkflowState):
    """
    Wykonuje finalny kod do przetwarzania danych.
    """
    print("--- WĘZEŁ: WYKONANIE KODU DANYCH  ---")
    try:
        print("  [INFO] Uruchamiam ostatecznie zatwierdzony kod...")
        
        # Definiujemy środowisko wykonawcze tylko z niezbędnymi bibliotekami
        exec_scope = {
            'pd': pd,
            'input_path': state['input_path'],
            'output_path': state['output_path']
        }
        
        exec(state['generated_code'], exec_scope)
        
        print("  [WYNIK] Kod wykonany pomyślnie.")
        return {"error_message": None, "correction_attempts": 0}
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"  [BŁĄD] Wystąpił błąd. Przekazywanie do inteligentnego debuggera:\n{error_traceback}")
        
        #--pamięć długotrwała: zapis błędu, sesja tymczasowa
        
        pending_session = {
            "initial_error": error_traceback,
            "initial_code": state['generated_code'],
            "fix_attempts": []  # Pusta lista na przyszłe próby naprawy
        }
        #--koniec--
        
        return {
            "failing_node": "data_code_executor", 
            "error_message": error_traceback, 
            "error_context_code": state['generated_code'], 
            "correction_attempts": state.get('correction_attempts', 0) + 1,
            "pending_fix_session": pending_session
        }

    
def universal_debugger_node(state: AgentWorkflowState):
    print(f"--- WĘZEŁ: INTELIGENTNY DEBUGGER (Błąd w: {state.get('failing_node')}) ---")
    failing_node_name = state.get('failing_node', 'unknown')
    
    
    
    MAIN_AGENT=state['config']['MAIN_AGENT']
    # llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    llm = ChatVertexAI(model_name=MAIN_AGENT,temperature=0.0, project=PROJECT_ID, location=LOCATION)
    tools = [propose_code_fix, request_package_installation, inspect_tool_code]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = PromptFactory.for_universal_debugger(
    failing_node=failing_node_name,
    error_message=state['error_message'],
    code_context=state['error_context_code'],
    active_policies=state.get("active_policies")
    )
    
    error_context = f"Wadliwy Kontekst:\n```\n{state['error_context_code']}\n```\n\nBłąd:\n```\n{state['error_message']}\n```"
    response = llm_with_tools.invoke(prompt + error_context)
    if not response.tool_calls:
        print("  [BŁĄD DEBUGGERA] Agent nie wybrał żadnego narzędzia. Eskalacja.")
        return {"error_message": "Debugger nie był w stanie podjąć decyzji.", "failing_node": "universal_debugger"}
    chosen_tool = response.tool_calls[0]
    tool_name = chosen_tool['name']
    tool_args = chosen_tool['args']
    print(f"  [DIAGNOZA] Debugger wybrał narzędzie: '{tool_name}' z argumentami: {tool_args}")
    return {"tool_choice": tool_name, "tool_args": tool_args, "debugger_analysis": tool_args.get("analysis", "")}


def apply_code_fix_node(state: AgentWorkflowState):
    """
    Aplikuje poprawkę kodu w sposób generyczny, używając centralnej mapy artefaktów.
    """
    print("--- WĘZEŁ: APLIKOWANIE POPRAWKI KODU (WERSJA ZMAPOWANA) ---")
    
    analysis = state.get("debugger_analysis", "")
    corrected_code = state.get("tool_args", {}).get("corrected_code")
    failing_node = state.get("failing_node")
    
    if not corrected_code:
        # ... (logika wymuszania generacji kodu - bez zmian)
        # ...
        pass # Ta część jest OK
        
    update = {
        "error_message": None, 
        "tool_choice": None, 
        "tool_args": None
    }

    # ZMIANA: Inteligentna logika oparta na mapie
    # Szukamy w mapie, który klucz w stanie należy zaktualizować.
    # Domyślnie, jeśli węzła nie ma na mapie, zakładamy, że dotyczy głównego kodu.
    target_state_key = CODE_ARTIFACT_MAP.get(failing_node, "generated_code")
    
    print(f"  [INFO] Błąd w węźle '{failing_node}'. Aplikowanie poprawki do artefaktu w stanie: '{target_state_key}'.")
    update[target_state_key] = corrected_code

    # Logika sesji naprawczej pozostaje bez zmian
    session = state.get('pending_fix_session')
    if not session:
        session = {
            "initial_error": state.get("error_message", "Brak błędu początkowego."),
            "initial_code": state.get("error_context_code", "Brak kodu początkowego."),
            "fix_attempts": []
        }

    attempt_info = {
        "debugger_analysis": analysis,
        "corrected_code": corrected_code,
        "attempt_number": len(session.get("fix_attempts", [])) + 1
    }
    
    if "fix_attempts" not in session:
        session["fix_attempts"] = []
    session["fix_attempts"].append(attempt_info)
    
    print(f"  [INFO] Dodano próbę naprawy nr {attempt_info['attempt_number']} do sesji.")
    
    update["pending_fix_session"] = session
    return update


def human_approval_node(state: AgentWorkflowState):
    print("\n" + "="*80 + "\n### WYMAGANA AKCJA CZŁOWIEKA  ###\n" + "="*80)
    package_name = state.get("tool_args", {}).get("package_name")
    user_input = input(f"Agent chce zainstalować pakiet '{package_name}'. Czy zgadzasz się? [y/n]: ").lower().strip()
    if user_input == 'y':
        print("Zgoda. Przechodzenie do instalacji.")
        return {"user_approval_status": "APPROVED", "package_to_install": package_name}
    else:
        print("Odrzucono. Przekazywanie do debuggera w celu znalezienia alternatywy.")
        new_error_message = f"Instalacja pakietu '{package_name}' została odrzucona przez użytkownika. Zmodyfikuj kod, aby nie używał tej zależności."
        return {"user_approval_status": "REJECTED", "error_message": new_error_message}


def package_installer_node(state: AgentWorkflowState):
    """Instaluje lub aktualizuje pakiet po uzyskaniu zgody."""
    package_name = state.get("package_to_install")
    
    # Domyślnie próbujemy aktualizacji, bo to rozwiązuje problemy z zależnościami
    success = install_package(package_name, upgrade=True)
    
    if success:
        return {"package_to_install": None, "user_approval_status": None, "error_message": None}
    else:
        return {"error_message": f"Operacja na pakiecie '{package_name}' nie powiodła się.", "failing_node": "package_installer"}


    
def summary_analyst_node(state: AgentWorkflowState) -> Dict[str, str]:
    """
    Agent, którego jedynym zadaniem jest analiza i stworzenie podsumowania tekstowego w HTML.
    """
    print("--- WĘZEŁ: ANALITYK PODSUMOWANIA ---")
    try:
        # Krok 1: Przygotuj dane wejściowe dla promptu
        df_original = pd.read_csv(state['input_path'])
        df_processed = pd.read_csv(state['output_path'])

        original_info_buf = io.StringIO()
        df_original.info(buf=original_info_buf)
        processed_info_buf = io.StringIO()
        df_processed.info(buf=processed_info_buf)

        original_summary = f"Podsumowanie danych ORYGINALNYCH:\n{df_original.describe().to_string()}\n{original_info_buf.getvalue()}"
        processed_summary = f"Podsumowanie danych PRZETWORZONYCH:\n{df_processed.describe().to_string()}\n{processed_info_buf.getvalue()}"

        # === POPRAWKA: Użycie dedykowanego promptu ===
        prompt = PromptFactory.for_summary_analyst(
        plan=state['plan'],
        original_summary=original_summary,
        processed_summary=processed_summary
        )
        
        llm = ChatAnthropic(model_name=state['config']['CODE_MODEL'], temperature=0.0, max_tokens=1024)
        structured_llm = llm.with_structured_output(ReportSummary)
        response = structured_llm.invoke(prompt)
        
        print("  [INFO] Analityk wygenerował podsumowanie HTML.")
        return {"summary_html": response.summary_html}
        
    except Exception as e:
        error_msg = f"Błąd w analityku podsumowania: {traceback.format_exc()}"
        print(f"  [BŁĄD] {error_msg}")
        return {
            "error_message": error_msg,
            "failing_node": "summary_analyst_node",
            "error_context_code": state.get('plan', 'Brak planu w stanie do analizy.'),
            "correction_attempts": state.get("correction_attempts", 0) + 1  # <-- DODAJ TĘ LINIĘ
        }


def plot_generator_node(state: AgentWorkflowState) -> Dict[str, str]:
    """
    Agent, którego jedynym zadaniem jest wygenerowanie KODU do tworzenia wykresów.
    """
    print("--- WĘZEŁ: GENERATOR WIZUALIZACJI ---")
    try:
        # --- NOWY KROK: POBIERZ AKTUALNE KOLUMNY Z PRZETWORZONEGO PLIKU ---
        df_processed_cols = pd.read_csv(state['output_path'], nrows=0).columns.tolist()

        # Przekaż aktualne kolumny do promptu, aby agent wiedział, na czym pracuje
        prompt = PromptFactory.for_plot_generator(
        plan=state['plan'],
        available_columns=df_processed_cols
        )
        
        MAIN_AGENT = state['config']['MAIN_AGENT']
        llm = ChatVertexAI(model_name=MAIN_AGENT, temperature=0.0, project=PROJECT_ID, location=LOCATION)
        structured_llm = llm.with_structured_output(PlottingCode)
        
        response = structured_llm.invoke(prompt)
        cleaned_code = extract_python_code(response.code)
        
        print("  [INFO] Generator stworzył kod do wizualizacji.")
        return {"plot_generation_code": cleaned_code}
        

    except Exception as e:
        error_msg = f"Błąd w generatorze wizualizacji: {traceback.format_exc()}"
        print(f"  [BŁĄD] {error_msg}")
        return {
            "error_message": error_msg,
            "failing_node": "plot_generator_node",
            "error_context_code": state.get('plan', 'Brak planu w stanie do analizy.'),
            "correction_attempts": state.get("correction_attempts", 0) + 1
        }


def report_composer_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Węzeł, który składa podsumowanie i wykresy w finalny raport HTML. Nie używa LLM.
    """
    print("--- WĘZEŁ: KOMPOZYTOR RAPORTU ---")
    try:
        summary_html = state.get("summary_html")
        plot_code = state.get("plot_generation_code")
        
        if not summary_html or not plot_code:
            raise ValueError("Brak podsumowania lub kodu do generowania wykresów w stanie.")

        # 1. Przygotuj środowisko wykonawcze dla kodu z wykresami
        exec_scope = {
            'pd': pd,
            'plt': plt,
            'df_original': pd.read_csv(state['input_path']),
            'df_processed': pd.read_csv(state['output_path']),
            'figures_to_embed': []
        }

        # 2. Wykonaj kod od agenta, aby wygenerować obiekty figur
        exec(plot_code, exec_scope)
        figures = exec_scope['figures_to_embed']
        print(f"  [INFO] Wykonano kod i wygenerowano {len(figures)} wykres(y).")

        # 3. Skonwertuj figury na tagi <img> z base64
        figures_html = ""
        for i, fig in enumerate(figures):
            figures_html += f"<h3>Wykres {i+1}</h3>{embed_plot_to_html(fig)}"

        # 4. Złóż finalny raport HTML
        final_html = f"""
        <!DOCTYPE html>
        <html lang="pl">
        <head>
            <meta charset="UTF-8">
            <title>Raport z Analizy Danych</title>
            <style>
                body {{ font-family: sans-serif; margin: 2em; background-color: #f9f9f9; }}
                .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 2em; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;}}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 4px; border-radius: 4px; margin-top: 1em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Raport z Przetwarzania Danych</h1>
                {summary_html}
                <h2>Wizualizacje</h2>
                {figures_html}
            </div>
        </body>
        </html>
        """

        # 5. Zapisz plik
        with open(state['report_output_path'], 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"✅ Raport został pomyślnie wygenerowany w {state['report_output_path']}")
        return {}

    except Exception as e:
        error_msg = f"Błąd w kompozytorze raportu: {traceback.format_exc()}"
        print(f"  [BŁĄD] {error_msg}")
        return {
            "error_message": error_msg, 
            "failing_node": "report_composer_node",
            "error_context_code": state.get("plot_generation_code", "Brak kodu do analizy."),
            "correction_attempts": state.get("correction_attempts", 0) + 1
        }
    
    
    
    

    
def sync_report_code_node(state: AgentWorkflowState):
    """Synchronizuje naprawiony kod z powrotem do stanu agenta raportującego."""
    print("--- WĘZEŁ: SYNCHRONIZACJA KODU RAPORTU ---")
    corrected_code = state.get("generated_code")
    return {"generated_report_code": corrected_code}   
    
    
    
def pre_audit_summarizer_node(state: AgentWorkflowState) -> dict:
    """
    Nowy węzeł, który streszcza duże artefakty PRZED przekazaniem ich do audytora.
    Działa jak "filtr" redukujący liczbę tokenów.
    """
    print("\n--- WĘZEŁ: Streszczanie Artefaktów Przed Audytem ---")
    
    summaries = {}
    artefacts_to_summarize = {
        "source_code": state.get('source_code', ''),
        "autogen_log": state.get('autogen_log', ''),
        "langgraph_log": state.get('langgraph_log', '')
    }
    MAIN_AGENT = state['config']['MAIN_AGENT']
    
    llm = ChatVertexAI(model_name=MAIN_AGENT, temperature=0.0)
    structured_llm = llm.with_structured_output(ArtefactSummary)

    for name, content in artefacts_to_summarize.items():
        if not content:
            summaries[f"{name}_summary"] = "Brak danych."
            continue
        
        print(f"  [INFO] Streszczanie artefaktu: {name}...")
        try:
            prompt = PromptFactory.for_artefact_summarizer(artefact_type=name, content=content)
            response = structured_llm.invoke(prompt)
            summaries[f"{name}_summary"] = response.summary
            print(f"  [SUKCES] Ukończono streszczenie: {name}.")
        except Exception as e:
            print(f"  [BŁĄD] Nie udało się zestreszczyć {name}: {e}")
            summaries[f"{name}_summary"] = "Błąd podczas streszczania."
            
    return summaries    
    
    

def meta_auditor_node(state: AgentWorkflowState):
    """Uruchamia audytora ORAZ zapisuje wspomnienia o sukcesie i wnioski META."""
    print("\n" + "="*80 + "\n### ### FAZA 3: META-AUDYT I KONSOLIDACJA WIEDZY ### ###\n" + "="*80 + "\n")
    
    try:
        
        CRITIC_MODEL=state['config']['CRITIC_MODEL']
        
        
        source_code_summary = state.get('source_code_summary', 'Brak podsumowania kodu.')
        autogen_log_summary = state.get('autogen_log_summary', 'Brak podsumowania logu planowania.')
        langgraph_log_summary = state.get('langgraph_log_summary', 'Brak podsumowania logu wykonania.')
        
        # Pozostałe dane skracamy dla bezpieczeństwa
        final_code = intelligent_truncate(state.get('generated_code', 'Brak kodu'), 1000)
        
        
        
        
        escalation_report_content = None
        escalation_path = state.get("escalation_report_path")
        if escalation_path:
            print(f"  [INFO] Wykryto raport z eskalacji. Wczytywanie pliku: {escalation_path}")
            try:
                with open(escalation_path, 'r', encoding='utf-8') as f:
                    escalation_report_content = f.read()
            except Exception as e:
                print(f"  [OSTRZEŻENIE] Nie udało się wczytać pliku z eskalacją: {e}")
        
        
        
        # ... (cała logika generowania raportu audytora, tak jak w oryginale)
        # Załóżmy, że wynikiem jest zmienna 'audit_report'
        
        
        final_report_summary = state.get("summary_html", "Brak podsumowania raportu.")
        
        # final_report_content = "Brak raportu do analizy."
        # try:
        #     with open(state['report_output_path'], 'r', encoding='utf-8') as f:
        #         final_report_content = f.read()
        # except Exception: pass
        
        
        # Logowanie rozmiaru każdego komponentu
        print(f"    - Rozmiar podsumowania kodu źródłowego: {len(source_code_summary)} znaków")
        print(f"    - Rozmiar podsumowania logu AutoGen:     {len(autogen_log_summary)} znaków")
        print(f"    - Rozmiar podsumowania logu LangGraph:  {len(langgraph_log_summary)} znaków")
        print(f"    - Rozmiar finalnego kodu:               {len(final_code)} znaków")
        print(f"    - Rozmiar raportu HTML:                 {len(final_report_summary)} znaków")
        if escalation_report_content:
            print(f"    - Rozmiar raportu eskalacji:          {len(escalation_report_content)} znaków")
        
        
        
        
        llm = ChatAnthropic(model_name=CRITIC_MODEL, temperature=0.0, max_tokens=2048)
        structured_llm = llm.with_structured_output(AuditReport)
        
        
        
        
        prompt = PromptFactory.for_meta_auditor(
            source_code=source_code_summary,
            autogen_log=autogen_log_summary,
            langgraph_log=langgraph_log_summary,
            final_code=final_code,
            final_report=final_report_summary,
            escalation_report=escalation_report_content
        )
        
        print(f"  [AUDYT-DIAGNOSTYKA] Całkowity rozmiar promptu wysyłanego do audytora: {len(prompt)} znaków")
        report_object = structured_llm.invoke(prompt)
        # ... (zapis raportu do pliku)

        audit_report = f"""
1.  **Ocena Planowania:**
    {report_object.planning_evaluation}

2.  **Ocena Wykonania:**
    {report_object.execution_evaluation}

3.  **Ocena Jakości Promptów (Analiza Meta):**
    {report_object.prompt_quality_analysis}

4.  **Rekomendacje do Samodoskonalenia:**
    {report_object.recommendations}
"""
        
        try:
            audit_report_path = "reports/meta_audit_report.txt"
            print(f"  [INFO] Zapisywanie raportu z audytu do: {audit_report_path}")
            with open(audit_report_path, "w", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write("### RAPORT Z META-AUDYTU SYSTEMU AI ###\n")
                f.write("="*50 + "\n\n")
                f.write(audit_report)
            print(f"  [SUKCES] Pomyślnie zapisano raport z audytu.")
        except Exception as e:
            print(f"  [BŁĄD] Nie udało się zapisać raportu z audytu: {e}")
        
        # 3. WYGENERUJ I ZAPISZ WNIOSEK META
        meta_insight_content = generate_meta_insight(audit_report)
        return {"meta_insight_content": meta_insight_content}

    except Exception as e:
        # ZMIANA: Bardziej szczegółowa obsługa błędu
        error_message = f"BŁĄD KRYTYCZNY podczas meta-audytu: {e}"
        print(error_message)
        # Dodatkowy log, jeśli błąd dotyczy długości promptu
        if "prompt is too long" in str(e) or "rate_limit_error" in str(e):
            print("\n  [DIAGNOZA] Błąd wskazuje na przekroczenie limitu tokenów. Sprawdź powyższe rozmiary "
                  "poszczególnych komponentów, aby zidentyfikować źródło problemu. "
                  "Prawdopodobnie węzeł 'pre_audit_summarizer_node' nie zredukował wystarczająco danych.")
        
        return {"meta_insight_content": None}

    
    
def human_escalation_node(state: AgentWorkflowState):
    """Węzeł eskalacji (bez zmian)."""
    print("\n==================================================")
    print(f"--- WĘZEŁ: ESKALACJA DO CZŁOWIEKA---")
    print("==================================================")
    # ... (reszta kodu bez zmian)
    report_content = f"""
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Problem: Przekroczono maksymalny limit ({MAX_CORRECTION_ATTEMPTS}) prób automatycznej naprawy.

Ostatnia analiza debuggera:
{state.get('debugger_analysis', 'Brak analizy.')}

Ostatni kod, który zawiódł:
```python
{state.get('error_context_code', 'Brak kodu.')}
```

Pełny traceback ostatniego błędu:
{state.get('error_message', 'Brak błędu.')}
"""
    file_name = f"reports/human_escalation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_name, "w", encoding="utf-8") as f: f.write(report_content)
    print(f"  [INFO] Raport dla człowieka został zapisany w pliku: {file_name}")
    return {"escalation_report_path": file_name}




def memory_consolidation_node(state: AgentWorkflowState):
    """
    Finalny węzeł grafu, odpowiedzialny za analizę całego przebiegu
    i zapisanie odpowiedniego rodzaju wspomnienia do pamięci długotrwałej.
    """
    print("\n" + "="*80 + "\n### ### FAZA 4: KONSOLIDACJA WIEDZY W PAMIĘCI ### ###\n" + "="*80 + "\n")
    memory_client = state['memory_client']
    run_id = state['run_id']
    dataset_signature = state['dataset_signature']
    
    # Scenariusz 1: Nastąpiła udana naprawa błędu w trakcie procesu.
    fix_session = state.get('pending_fix_session')
    if fix_session and fix_session.get("fix_attempts"):
        print("  [PAMIĘĆ] Wykryto udaną sesję naprawczą. Zapisuję wspomnienie typu SUCCESSFUL_FIX.")
        distilled_content = distill_fix_memory(
            initial_error=fix_session['initial_error'],
            fix_attempts=fix_session['fix_attempts'],
            successful_code=state['generated_code'] # lub inny odpowiedni kod
        )
        record = MemoryRecord(
            run_id=run_id,
            memory_type=MemoryType.SUCCESSFUL_FIX,
            dataset_signature=dataset_signature,
            source_node="memory_consolidation_node",
            content=distilled_content,
            metadata={"total_attempts": len(fix_session['fix_attempts'])}
        )
        memory_client.add_memory(record)

    # Scenariusz 2: Proces zakończył się pełnym sukcesem bez żadnych błędów.
    elif not state.get("escalation_report_path"):
        print("  [PAMIĘĆ] Wykryto pomyślny przebieg bez błędów. Zapisuję wspomnienie typu SUCCESSFUL_WORKFLOW.")
        distilled_content = distill_successful_workflow(
            plan=state.get('plan', ''),
            final_code=state.get('generated_code', ''),
            langgraph_log=state.get('langgraph_log', '')
        )
        record = MemoryRecord(
            run_id=run_id,
            memory_type=MemoryType.SUCCESSFUL_WORKFLOW,
            dataset_signature=dataset_signature,
            source_node="memory_consolidation_node",
            content=distilled_content,
            metadata={"importance_score": 0.9}
        )
        memory_client.add_memory(record)

    # Scenariusz 3: Nastąpiła eskalacja do człowieka lub inny nieprzewidziany błąd.
    else:
        print("  [PAMIĘĆ] Wykryto eskalację lub nieudany przebieg. Nie zapisuję wspomnienia o sukcesie.")
        # W przyszłości można tu dodać logikę zapisu wspomnienia o porażce.

    # Na samym końcu zapisujemy wniosek z audytu, jeśli istnieje
    meta_insight_content = state.get("meta_insight_content") # Zakładamy, że audytor umieścił to w stanie
    if meta_insight_content:
        print("  [PAMIĘĆ] Zapisuję wniosek META z audytu.")
        insight_record = MemoryRecord(
            run_id=run_id, memory_type=MemoryType.META_INSIGHT,
            dataset_signature=dataset_signature, source_node="meta_auditor_node",
            content=meta_insight_content, metadata={"importance_score": 1.0}
        )
        memory_client.add_memory(insight_record)
        
    return {}
