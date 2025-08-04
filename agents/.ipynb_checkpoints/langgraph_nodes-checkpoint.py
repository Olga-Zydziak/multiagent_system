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
from config import MAX_CORRECTION_ATTEMPTS, PROJECT_ID,LOCATION
from memory.memory_utils import *
from memory.memory_models import *
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
    prompt = LangchainAgentsPrompts.code_generator(state['plan'], state['available_columns'])
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
    failing_node_name = state.get('failing_node', 'unknown')
    
    
    
    MAIN_AGENT=state['config']['MAIN_AGENT']
    # llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    llm = ChatVertexAI(model_name=MAIN_AGENT,temperature=0.0, project=PROJECT_ID, location=LOCATION)
    tools = [propose_code_fix, request_package_installation, inspect_tool_code]
    llm_with_tools = llm.bind_tools(tools)
    prompt = LangchainAgentsPrompts.tool_based_debugger(failing_node=failing_node_name,active_policies=state.get("active_policies"))
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
    """Aplikuje poprawkę kodu zaproponowaną przez debuggera."""
    print("--- WĘZEŁ: APLIKOWANIE POPRAWKI KODU ---")
    
    CODE_MODEL=state['config']['CODE_MODEL']
    
    analysis = state.get("debugger_analysis", "")
    corrected_code = state.get("tool_args", {}).get("corrected_code")
    failing_node = state.get("failing_node")
    
    
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
    
    update = {
        "error_message": None, 
        "tool_choice": None, 
        "tool_args": None
    }

    if failing_node == "plot_generator_node":
        print("  [INFO] Aplikowanie poprawki do kodu generującego wykresy.")
        update["plot_generation_code"] = corrected_code
    elif failing_node == "summary_analyst_node":
        print("  [INFO] Aplikowanie poprawki do podsumowania HTML.")
        update["summary_html"] = corrected_code
    else: # Domyślnie traktuj jako główny kod
        print("  [INFO] Aplikowanie poprawki do głównego kodu przetwarzania danych.")
        update["generated_code"] = corrected_code

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
        prompt = LangchainAgentsPrompts.summary_analyst_prompt(
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
        prompt = LangchainAgentsPrompts.plot_generator_prompt(
            plan=state['plan'],
            available_columns=df_processed_cols  # Przekazanie nowej informacji
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
    
    
def meta_auditor_node(state: AgentWorkflowState):
    """Uruchamia audytora ORAZ zapisuje wspomnienia o sukcesie i wnioski META."""
    print("\n" + "="*80 + "\n### ### FAZA 3: META-AUDYT I KONSOLIDACJA WIEDZY ### ###\n" + "="*80 + "\n")
    memory_client = state['memory_client']

    if not state.get("escalation_report_path"):
        try:
            print("  [INFO] Uruchamiam proces destylacji wspomnienia o sukcesie...")
            # --- ZMIANA 2: Zabezpieczenie przed limitem tokenów ---
            truncated_plan = intelligent_truncate(state.get('plan', ''), 3000)
            distilled_content = distill_success_memory(final_plan=truncated_plan)
            
            if distilled_content and distilled_content.get("key_insight"):
                # Dodatkowe zabezpieczenie dla wyniku destylacji
                distilled_content["key_insight"] = intelligent_truncate(distilled_content["key_insight"], 2000)

                plan_record = MemoryRecord(
                    run_id=state['run_id'],
                    memory_type=MemoryType.SUCCESSFUL_PLAN,
                    dataset_signature=state['dataset_signature'],
                    source_node="meta_auditor_node",
                    content=distilled_content,
                    metadata={"importance_score": 0.8}
                )
                memory_client.add_memory(plan_record)
        except Exception as e:
            print(f"  [BŁĄD ZAPISU PAMIĘCI] Nie udało się zapisać udanego planu: {e}")
    
    # 2. Uruchom audytora (logika bez zmian)
    try:
        
        CRITIC_MODEL=state['config']['CRITIC_MODEL']
        
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
        final_report_content = "Brak raportu do analizy."
        try:
            with open(state['report_output_path'], 'r', encoding='utf-8') as f:
                final_report_content = f.read()
        except Exception: pass
        
        llm = ChatAnthropic(model_name=CRITIC_MODEL, temperature=0.0, max_tokens=2048)
        prompt = LangchainAgentsPrompts.create_meta_auditor_prompt(
            source_code=intelligent_truncate(state['source_code'], 8000),
            autogen_conversation=intelligent_truncate(state['autogen_log'], 6000),
            langgraph_log=intelligent_truncate(state.get('langgraph_log', ''), 6000),
            final_code=state.get('generated_code', 'Brak kodu'),
            final_report=final_report_content,
            escalation_report=escalation_report_content
        )
        audit_report = llm.invoke(prompt).content
        # ... (zapis raportu do pliku)

        
        
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
