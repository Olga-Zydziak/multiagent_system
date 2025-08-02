import os
import io
import sys
import subprocess
import tempfile
import traceback
import uuid
import json
import re
from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
import pandas as pd
import langchain
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from .state import AgentWorkflowState
from prompts import Langchain_Agents_prompts
from tools.utils import *
from tools.langchain_tools import *
from prompts import ArchitecturalRule, ArchitecturalRulesManager,ARCHITECTURAL_RULES
from config import MAX_CORRECTION_ATTEMPTS, PROJECT_ID,LOCATION
# --- Definicje węzłów LangGraph ---

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
    print("---  WĘZEŁ: GENERATOR KODU ---")
    
    
    CODE_MODEL=state['config']['CODE_MODEL']
    
    llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    prompt = Langchain_Agents_prompts.code_generator(state['plan'], state['available_columns'])
    response = llm.invoke(prompt).content
    code = extract_python_code(response)
    
    print("\nAgent-Analityk wygenerował następujący kod:")
    print("--------------------------------------------------")
    print(code)
    print("--------------------------------------------------")
    return {"generated_code": code}


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
    MAIN_AGENT=state['config']['MAIN_AGENT']
    # llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    llm = ChatVertexAI(model_name=MAIN_AGENT,temperature=0.0, project=PROJECT_ID, location=LOCATION)
    tools = [propose_code_fix, request_package_installation]
    llm_with_tools = llm.bind_tools(tools)
    prompt = Langchain_Agents_prompts.tool_based_debugger()
    error_context = f"Wadliwy Kod:\n```python\n{state['error_context_code']}\n```\n\nBłąd:\n```\n{state['error_message']}\n```"
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
    """Aplikuje poprawkę kodu zaproponowaną przez debuggera."""
    print("--- WĘZEŁ: APLIKOWANIE POPRAWKI KODU ---")
    
    CODE_MODEL=state['config']['CODE_MODEL']
    
    analysis = state.get("debugger_analysis", "")
    corrected_code = state.get("tool_args", {}).get("corrected_code")
    
    if not corrected_code:
        print("  [OSTRZEŻENIE] Debugger nie dostarczył kodu. Wymuszam jego wygenerowanie...")
        
        # Tworzymy bardzo prosty prompt, który ma tylko jedno zadanie
        force_prompt = f"""Na podstawie poniższej analizy i wadliwego kodu, wygeneruj PEŁNY, POPRAWIONY i gotowy do uruchomienia skrypt Pythona.
        Twoja odpowiedź musi zawierać TYLKO i WYŁĄCZNIE blok kodu, bez żadnych dodatkowych wyjaśnień.

        [ANALIZA BŁĘDU]:
        {analysis}

        [WADLIWY KOD]:
        ```python
        {state['error_context_code']}"""
        
        
        try:
            llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
            response = llm.invoke(force_prompt).content
            corrected_code = extract_python_code(response) # Używamy istniejącej funkcji pomocniczej
            print("  [INFO] Pomyślnie wymuszono wygenerowanie kodu.")
        except Exception as e:
            print(f"  [BŁĄD KRYTYCZNY] Nie udało się wymusić generacji kodu: {e}")
            return {"error_message": "Nie udało się naprawić kodu nawet po eskalacji."}
        
        
    #--pamięć długotrwała info dla pamieci--
    
    
    session = state.get('pending_fix_session')
    if not session:
        # Sytuacja awaryjna, nie powinno się zdarzyć w normalnym przepływie
        print("  [OSTRZEŻENIE] Próba aplikacji poprawki bez aktywnej sesji naprawczej.")
        session = {}

    # Dodajemy informacje o tej konkretnej próbie do listy w sesji
    attempt_info = {
        "debugger_analysis": state.get("debugger_analysis", "Brak analizy."),
        "corrected_code": corrected_code,
        "attempt_number": len(session.get("fix_attempts", [])) + 1
    }
    
    if "fix_attempts" in session:
        session["fix_attempts"].append(attempt_info)
    else:
        session["fix_attempts"] = [attempt_info]
    
    print(f"  [INFO] Dodano próbę naprawy nr {attempt_info['attempt_number']} do sesji.")
    
    
    #--koniec--
    
    return {
        "generated_code": corrected_code, 
        "error_message": None, 
        "tool_choice": None, 
        "tool_args": None,
        "pending_fix_session": session  # Aktualizujemy sesję w stanie
    }


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

def commit_memory_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Zapisuje skonsolidowaną wiedzę do pamięci po udanej naprawie kodu."""
    session = state.get('pending_fix_session')
    
    # Jeśli nie ma sesji (np. kod zadziałał za 1. razem), nie rób nic
    if not session or not session.get("fix_attempts"):
        return {"pending_fix_session": None}

    print("--- WĘZEŁ: ZATWIERDZANIE WIEDZY W PAMIĘCI ---")
    
    distilled_content = distill_full_fix_session(
        initial_error=session['initial_error'],
        fix_attempts=session['fix_attempts'],
        successful_code=state['generated_code']
    )
    
    memory_client = state['memory_client']
    final_record = MemoryRecord(
        run_id=state['run_id'],
        memory_type=MemoryType.SUCCESSFUL_FIX, # Teraz to jest prawdziwy sukces
        dataset_signature=state['dataset_signature'],
        source_node="commit_memory_node",
        content=distilled_content,
        metadata={"total_attempts": len(session['fix_attempts'])}
    )
    memory_client.add_memory(final_record)
    
    # Wyczyść sesję po udanym zapisie
    return {"pending_fix_session": None} 

    
    
def reporting_agent_node(state: AgentWorkflowState):
    """
    Wczytuje dane wejściowe i przetworzone, tworzy ich podsumowania statystyczne,
    a następnie wywołuje agenta w celu wygenerowania kodu analitycznego.
    """
    print("\n--- WĘZEŁ: AGENT RAPORTUJĄCY (ANALIZA DANYCH I GENEROWANIE KODU) ---")
    
    try:
        
        CODE_MODEL=state['config']['CODE_MODEL']
        
        # --- NOWY KROK: Wczytanie i analiza danych ---
        print("  [INFO] Wczytywanie danych do analizy porównawczej...")
        df_original = pd.read_csv(state['input_path'])
        df_processed = pd.read_csv(state['output_path'])

        # Tworzenie zwięzłych podsumowań tekstowych dla LLM
        # Używamy io.StringIO, aby przechwycić 'print' z df.info() do stringa
        original_info_buf = io.StringIO()
        df_original.info(buf=original_info_buf)
        
        processed_info_buf = io.StringIO()
        df_processed.info(buf=processed_info_buf)

        original_summary = f"""
### Podsumowanie danych ORYGINALNYCH ###
Pierwsze 3 wiersze:
{df_original.head(3).to_string()}

Informacje o kolumnach:
{original_info_buf.getvalue()}
Statystyki (dla kolumn numerycznych):
{df_original.describe().to_string()}
"""
        processed_summary = f"""
### Podsumowanie danych PRZETWORZONYCH ###
Pierwsze 3 wiersze:
{df_processed.head(3).to_string()}

Informacje o kolumnach:
{processed_info_buf.getvalue()}
Statystyki (dla kolumn numerycznych):
{df_processed.describe().to_string()}
"""
        print("  [INFO] Podsumowania danych wygenerowane.")
        # --- KONIEC NOWEGO KROKU ---

        # Krok 2: Utwórz precyzyjny prompt z nowym kontekstem
        prompt = PromptTemplates.create_reporting_prompt(
            plan=state['plan'],
            original_summary=original_summary,
            processed_summary=processed_summary
        )
        
        # Krok 3: Wywołaj LLM (bez zmian)
        llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
        structured_llm = llm.with_structured_output(GeneratedPythonScript)
        response_object = structured_llm.invoke(prompt)
        report_analysis_code = response_object.script_code
        
        print("  [INFO] Agent-Analityk wygenerował kod analityczny na podstawie danych.")
        
        return {"generated_report_code": report_analysis_code}

    except Exception as e:
        print(f"  [BŁĄD] Krytyczny błąd w agencie raportującym: {traceback.format_exc()}")
        return {"generated_report_code": None}

def report_executor_node(state: AgentWorkflowState):
    """
    Wczytuje zewnętrzny szablon HTML, wykonuje kod analityczny od agenta,
    a następnie wstawia wyniki do szablonu, tworząc finalny raport.
    """
    print("--- WĘZEŁ: WYKONANIE KODU RAPORTU (Z ZEWNĘTRZNEGO SZABLONU) ---")
    analysis_code = state.get("generated_report_code")
    
    if not analysis_code:
        return {"error_message": "Brak kodu analitycznego do wykonania.", "failing_node": "report_executor"}

    try:
        # Krok 1: Zbuduj kompletny, wykonywalny skrypt do wygenerowania "ciała" raportu
        # Ten skrypt zawiera wszystkie potrzebne importy i funkcje pomocnicze.
        body_script_to_execute = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def embed_plot_to_html(figure):
    \"\"\"Konwertuje figurę matplotlib do stringa base64 do osadzenia w HTML.\"\"\"
    buffer = BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    plt.close(figure)
    return f'<img src="data:image/png;base64,{{graphic}}" alt="Wykres analizy danych" style="max-width: 100%; height: auto;"/>'

# --- Kod wygenerowany przez agenta-analityka ---
{analysis_code}
# ---------------------------------------------

# Przygotowanie finalnego "ciała" HTML do wstawienia w szablon
html_body_content = ""
if 'summary_text' in locals():
    html_body_content += "<h2>Podsumowanie</h2>" + summary_text
if 'figures_to_embed' in locals() and isinstance(figures_to_embed, list):
    html_body_content += "<h2>Wizualizacje</h2>"
    for fig in figures_to_embed:
        html_body_content += embed_plot_to_html(fig)
"""
        # Krok 2: Przygotuj środowisko i wykonaj powyższy skrypt, aby uzyskać treść raportu
        print("  [INFO] Wykonywanie kodu analitycznego w celu wygenerowania treści raportu...")
        exec_scope = {
            'df_original': pd.read_csv(state['input_path']),
            'df_processed': pd.read_csv(state['output_path']),
        }
        exec(body_script_to_execute, exec_scope)
        generated_html_body = exec_scope['html_body_content']

        # Krok 3: Wczytaj zewnętrzny szablon HTML
        print("  [INFO] Wczytywanie szablonu z pliku report_template.html...")
        with open("report_template.html", "r", encoding="utf-8") as f:
            template = f.read()

        # Krok 4: Wstaw wygenerowaną treść do szablonu i zapisz finalny raport
        final_html = template.format(generated_html_body=generated_html_body)
        with open(state['report_output_path'], 'w', encoding='utf-8') as f:
            f.write(final_html)

        print(f"  [INFO] Raport HTML został pomyślnie zapisany w: {state['report_output_path']}")
        return {"error_message": None} # Sukces

    except Exception:
        error_traceback = traceback.format_exc()
        print(f"  [BŁĄD] Wystąpił błąd podczas wykonywania skryptu raportu:\n{error_traceback}")
        # Przekazujemy do debuggera tylko ten fragment, który zawiódł (kod od agenta)
        return {
            "failing_node": "report_executor", 
            "error_message": error_traceback, 
            "error_context_code": analysis_code, 
            "correction_attempts": state.get('correction_attempts', 0) + 1
        }

    
def sync_report_code_node(state: AgentWorkflowState):
    """Synchronizuje naprawiony kod z powrotem do stanu agenta raportującego."""
    print("--- WĘZEŁ: SYNCHRONIZACJA KODU RAPORTU ---")
    corrected_code = state.get("generated_code")
    return {"generated_report_code": corrected_code}   
    
    
def meta_auditor_node(state: AgentWorkflowState):
    """Uruchamia audytora ORAZ zapisuje wspomnienia o sukcesie i wnioski META."""
    print("\n" + "="*80 + "\n### ### FAZA 3: META-AUDYT I KONSOLIDACJA WIEDZY ### ###\n" + "="*80 + "\n")
    memory_client = state['memory_client']

    # 1. Zapisz wspomnienie o udanym planie (jeśli nie było błędów)
    if state.get('plan') and not state.get('error_message'):
        distilled_content = distill_success_memory(final_plan=state['plan'])
        plan_record = MemoryRecord(
            run_id=state['run_id'], memory_type=MemoryType.SUCCESSFUL_PLAN,
            dataset_signature=state['dataset_signature'], source_node="meta_auditor_node",
            content=distilled_content, metadata={"importance_score": 0.8}
        )
        memory_client.add_memory(plan_record)
    
    # 2. Uruchom audytora (logika bez zmian)
    try:
        
        CRITIC_MODEL=state['config']['CRITIC_MODEL']
        
        # ... (cała logika generowania raportu audytora, tak jak w oryginale)
        # Załóżmy, że wynikiem jest zmienna 'audit_report'
        final_report_content = "Brak raportu do analizy."
        try:
            with open(state['report_output_path'], 'r', encoding='utf-8') as f:
                final_report_content = f.read()
        except Exception: pass
        
        llm = ChatAnthropic(model_name=CRITIC_MODEL, temperature=0.0, max_tokens=2048)
        prompt = Langchain_Agents_prompts.create_meta_auditor_prompt(
            source_code=state['source_code'], autogen_conversation=state['autogen_log'],
            langgraph_log=state['langgraph_log'], final_code=state.get('generated_code', 'Brak kodu'),
            final_report=final_report_content
        )
        audit_report = llm.invoke(prompt).content
        # ... (zapis raportu do pliku)

        # 3. WYGENERUJ I ZAPISZ WNIOSEK META
        meta_insight_content = generate_meta_insight(audit_report)
        if meta_insight_content:
            insight_record = MemoryRecord(
                run_id=state['run_id'], memory_type=MemoryType.META_INSIGHT,
                dataset_signature=state['dataset_signature'], source_node="meta_auditor_node",
                content=meta_insight_content, metadata={"importance_score": 1.0}
            )
            memory_client.add_memory(insight_record)

    except Exception as e:
        print(f"BŁĄD KRYTYCZNY podczas meta-audytu: {e}")
    return {}

    
    
def human_escalation_node(state: AgentWorkflowState):
    """Węzeł eskalacji (bez zmian)."""
    print("\n==================================================")
    print(f"--- WĘZEŁ: ESKALACJA DO CZŁOWIEKA---")
    print("==================================================")
    # ... (reszta kodu bez zmian)
    report_content = f"""
Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
    file_name = f"human_escalation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_name, "w", encoding="utf-8") as f: f.write(report_content)
    print(f"  [INFO] Raport dla człowieka został zapisany w pliku: {file_name}")
    return {}
