import os
import io
import sys
import subprocess
import inspect
import matplotlib.pyplot as plt
import base64 
import tempfile
import traceback
import uuid
import json
import re
from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
import pandas as pd
import datetime
import logging


#--funkcja dla pamieci--
def intelligent_truncate(text: str, max_len: int) -> str:
    """Skraca tekst, zachowując jego początek i koniec."""
    if not isinstance(text, str) or len(text) <= max_len:
        return text
    half_len = (max_len - 25) // 2
    start = text[:half_len]
    end = text[-half_len:]
    return f"{start}\n\n[... treść skrócona ...]\n\n{end}"

def extract_python_code(response: str) -> str:
    response = response.strip()
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match: return match.group(1).strip()
    if response.startswith("'''") and response.endswith("'''"): return response[3:-3].strip()
    if response.startswith('"""') and response.endswith('"""'): return response[3:-3].strip()
    return response


def install_package(package_name: str, upgrade: bool = True) -> bool:
    """
    Instaluje lub aktualizuje podany pakiet używając pip.
    
    Args:
        package_name (str): Nazwa pakietu do instalacji.
        upgrade (bool): Jeśli True, używa flagi --upgrade.
    """
    try:
        command = [sys.executable, "-m", "pip", "install", package_name]
        if upgrade:
            command.insert(2, "--upgrade")
        
        action = "Aktualizacja" if upgrade else "Instalacja"
        print(f"  [INSTALATOR] Próba: {action} pakietu {package_name}...")
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  [INSTALATOR] Pomyślnie zakończono. Logi pip:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [INSTALATOR] Błąd podczas operacji na pakiecie {package_name}.\n{e.stderr}")
        return False
    

    
#DLA report agenta
def embed_plot_to_html(figure) -> str:
    """Konwertuje figurę matplotlib do stringa base64 do osadzenia w HTML."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    plt.close(figure) # Ważne: zamykamy figurę
    return f'<img src="data:image/png;base64,{graphic}" alt="Wykres analizy danych"/>'



#Dla meta agenta
def read_source_code(file_path: str) -> str:
    """Odczytuje zawartość pliku kodu źródłowego."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: return f"Nie udało się odczytać kodu źródłowego: {e}"


def read_project_source_code(root_dir=".", exclude_dirs=None, exclude_files=None): # <-- Dodaj nowy argument
    """
    Rekursywnie skanuje katalog projektu, odczytuje zawartość plików .py i .ipynb,
    i łączy je w jeden, sformatowany string dla meta-audytora.
    """
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.ipynb_checkpoints', 'reports', '.git'}
    
    if exclude_files is None: # <-- Dodaj tę sekcję
        exclude_files = set()

    full_source_code = ""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            if filename in exclude_files: # <-- Dodaj tę linię, aby pomijać pliki
                continue
            
            if filename.endswith(".py") or filename.endswith(".ipynb"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        full_source_code += f"--- FILE: {file_path} ---\n\n{content}\n\n"
                except Exception as e:
                    full_source_code += f"--- FILE: {file_path} ---\n\n[BŁĄD ODCZYTU: {e}]\n\n"
                    
    return full_source_code

#Zapis planowania preprocessingu- AutoGen
def save_autogen_conversation_log(log_content: str, file_path: str):
    """Zapisuje pełną treść konwersacji agentów AutoGen do pliku tekstowego."""
    print(f"INFO: Próba zapisu pełnego logu rozmowy do pliku: {file_path}")
    try:
        # Upewniamy się, że katalog 'reports' istnieje
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("="*40 + "\n")
            f.write("### PEŁNY ZAPIS ROZMOWY AGENTÓW (FAZA PLANOWANIA) ###\n")
            f.write("="*40 + "\n\n")
            f.write(log_content)
            
        print(f"✅ SUKCES: Log rozmowy został pomyślnie zapisany.")
    except Exception as e:
        print(f"❌ BŁĄD: Nie udało się zapisać logu rozmowy. Przyczyna: {e}")


        
#Zapis rozmowy agentow wykonowczych- LangChain        
def save_langgraph_execution_log(log_content: str, file_path: str):
    """Zapisuje pełny, szczegółowy log z wykonania grafu LangGraph do pliku."""
    print(f"INFO: Próba zapisu pełnego logu wykonania LangGraph do pliku: {file_path}")
    try:
        # Upewniamy się, że katalog 'reports' istnieje
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("="*40 + "\n")
            f.write("### PEŁNY ZAPIS WYKONANIA GRAFU LANGGRAPH (FAZA WYKONANIA) ###\n")
            f.write("="*40 + "\n\n")
            f.write(log_content)
            
        print(f"✅ SUKCES: Log wykonania LangGraph został pomyślnie zapisany.")
    except Exception as e:
        print(f"❌ BŁĄD: Nie udało się zapisać logu LangGraph. Przyczyna: {e}")  
        
        
TOOL_REGISTRY = {
    "embed_plot_to_html": embed_plot_to_html
}