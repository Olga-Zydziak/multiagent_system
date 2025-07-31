#!/usr/bin/env python
# coding: utf-8

# ################################################################################
# ### IMPORT WSZYSTKICH POTRZEBNYCH ZALEŻNOŚCI
# ################################################################################

# In[1]:


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
import datetime
import logging
from enum import Enum
import matplotlib.pyplot as plt
# --- Frameworki Agentów ---
import autogen
from autogen import Agent, ConversableAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from google.cloud import secretmanager
from typing import Optional, Tuple
from langchain_anthropic import ChatAnthropic
import langchain
from langchain.cache import SQLiteCache

#--Frameworki pamięci--

from memory_models import MemoryRecord, MemoryType,DistilledMemory,DistilledSuccessMemory
from memory_bank_client import MemoryBankClient
import vertexai
from vertexai import agent_engines


# ################################################################################
# ### KONFIGURACJA PODSTAWOWA
# ################################################################################

# In[2]:


def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Pobiera wartość sekretu z Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
   
    return response.payload.data.decode("UTF-8")


# In[3]:


class ApiType(Enum):
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    def __str__(self):
        return self.value


# In[4]:


LOCATION="us-central1"
PROJECT_ID="dark-data-discovery"

#---------AGENTS--------:
MAIN_AGENT="gemini-2.5-pro"
API_TYPE_GEMINI=str(ApiType.GOOGLE)

CRITIC_MODEL="claude-3-7-sonnet-20250219"
CODE_MODEL="claude-sonnet-4-20250514"
API_TYPE_SONNET = str(ApiType.ANTHROPIC)

LANGCHAIN_API_KEY = get_secret(PROJECT_ID,"LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY=get_secret(PROJECT_ID,"ANTHROPIC_API_KEY")

MEMORY_ENGINE_DISPLAY_NAME="memory-gamma-way"

INPUT_FILE_PATH = "gs://super_model/data/structural_data/synthetic_fraud_dataset.csv"

MAX_CORRECTION_ATTEMPTS=5


# In[5]:


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Projekt Multi-Agent-System v9.0-Integrated"
os.environ["ANTHROPIC_API_KEY"] =ANTHROPIC_API_KEY


# In[6]:


#---cache-------
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


# ################################################################################
# ### INICJOWANIE I KONFIGURACJA PAMIĘCI DŁUGOTRWAŁEJ
# ################################################################################

# In[7]:


AGENT_ENGINE_NAME = "" # Zostanie wypełniona po pobraniu lub utworzeniu silnika

# Inicjalizacja głównego klienta Vertex AI
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

def get_or_create_agent_engine(display_name: str) :
    """
    Pobiera istniejący Agent Engine po nazwie wyświetlanej lub tworzy nowy, jeśli nie istnieje.
    """
    # 1. Pobierz listę wszystkich istniejących silników w projekcie
    all_engines = agent_engines.list()
    
    # 2. Sprawdź, czy któryś z nich ma pasującą nazwę
    for engine in all_engines:
        if engine.display_name == display_name:
            print(f"INFO: Znaleziono i połączono z istniejącym Agent Engine: '{display_name}'")
            return engine
            
    # 3. Jeśli pętla się zakończyła i nic nie znaleziono, stwórz nowy silnik
    print(f"INFO: Nie znaleziono Agent Engine o nazwie '{display_name}'. Tworzenie nowego...")
    try:
        new_engine = agent_engines.create(
            display_name=display_name
        )
        print(f"INFO: Pomyślnie utworzono nowy Agent Engine.")
        return new_engine
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: Nie można utworzyć Agent Engine. Sprawdź konfigurację i uprawnienia. Błąd: {e}")
        exit()


# In[8]:


agent_engine =get_or_create_agent_engine(MEMORY_ENGINE_DISPLAY_NAME)
AGENT_ENGINE_NAME = agent_engine.resource_name
print(AGENT_ENGINE_NAME)


# ################################################################################
# ### ### FAZA 1: PLANOWANIE STRATEGICZNE (AutoGen)
# ################################################################################

# In[9]:


#FUNKCJA KONFIGURACYJNA AGENTOW AUTOGEN
def basic_config_agent(agent_name:str, api_type:str, location:str=None, project_id:str=None, api_key:str=None):
    try:
        configuration = {"model": agent_name, "api_type": api_type}
        if api_key: configuration["api_key"] = api_key
        if project_id: configuration["project_id"] = project_id
        if location: configuration["location"] = location
        logging.info(f"Model configuration: {configuration}")
        return [configuration]

    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI or configure LLM: {e}")
        print(f"Error: Failed to initialize Vertex AI or configure LLM. Please check your project ID, region, and permissions. Details: {e}")
        exit()



#TRIGGER AGENT
class TriggerAgent(ConversableAgent):
    """Agent decydujący, czy dane nadają się do dalszego przetwarzania."""
    def __init__(self, llm_config):
        super().__init__(
            name="TriggerAgent",
            llm_config=llm_config,
            system_message="""Jesteś 'Strażnikiem Danych'. Twoim jedynym zadaniem jest analiza podsumowania danych (nazwy kolumn, pierwsze wiersze).
Na tej podstawie musisz podjąć decyzję: czy te dane mają charakter **tabularyczny** (jak plik CSV lub tabela bazy danych)?
- Jeśli TAK: odpowiedz **tylko i wyłącznie**: 'Dane są tabularyczne. Przekazuję do PlannerAgent w celu stworzenia planu analizy.'. Nie dodawaj nic więcej.
- Jeśli NIE (np. są to logi serwera, obrazy, czysty tekst): Twoja wiadomość MUSI kończyć się słowem 'TERMINATE'. Wyjaśnij krótko, dlaczego dane nie są tabularyczne, np. 'Dane nie są tabularyczne, to zbiór artykułów tekstowych. TERMINATE'."""
        )

#PLANNER AGENT        
class PlannerAgent(ConversableAgent):
    """Agent tworzący szczegółowy plan przygotowania danych."""
    def __init__(self, llm_config):
        super().__init__(
            name="PlannerAgent",
            llm_config=llm_config,
            system_message="""Jesteś 'Architektem Planu'. Otrzymałeś potwierdzenie, że dane są tabularyczne.
Twoim zadaniem jest stworzenie szczegółowego, numerowanego planu czyszczenia i przygotowania danych do ogólnej analizy i modelowania. Plan musi być praktyczny i zgodny z najlepszymi praktykami.
Twoje zadanie składa się z dwóch części:
1.  **Analiza Inspiracji:** Jeśli w wiadomości od użytkownika znajduje się sekcja '--- INSPIRACJE Z POPRZEDNICH URUCHOMIEŃ ---', 
potraktuj ją jako cenną inspirację i punkt wyjścia. Zawiera ona sprawdzoną strategię ("złotą myśl") i może również zawierać konkretne kroki. Twoim zadaniem jest **krytyczna adaptacja** tego planu. 
**Sprawdź, czy każdy krok z inspiracji ma sens w kontekście AKTUALNEGO podglądu danych.** Możesz usunąć, dodać lub zmodyfikować kroki, aby idealnie pasowały do obecnego problemu.
2.  **Tworzenie Planu:** Jeśli nie ma inspiracji, stwórz nowy, solidny plan od podstaw.
Plan powinien zawierać kroki takie jak:
1.  Weryfikacja i obsługa brakujących wartości (np. strategia imputacji dla każdej istotnej kolumny).
2.  Weryfikacja i korekta typów danych (np. konwersja stringów na daty lub liczby).
3.  Inżynieria cech (np. tworzenie nowych, użytecznych kolumn jak 'dzien_tygodnia' z daty lub kategoryzacja wartości liczbowych).
4.  Wykrywanie i obsługa wartości odstających (outlierów).
5.  Normalizacja lub skalowanie danych (jeśli to konieczne, wyjaśnij krótko dlaczego).

Po przedstawieniu pierwszej wersji planu, oczekuj na recenzję od CriticAgenta.
- Jeśli CriticAgent prześle uwagi, stwórz **NOWĄ, KOMPLETNĄ WERSJĘ** planu, która uwzględnia **WSZYSTKIE** jego sugestie.
- W poprawionym planie zaznacz, co zostało zmienione. Prześlij zaktualizowany plan z powrotem do CriticAgenta.
Kontynuuj ten proces, aż CriticAgent ostatecznie zaakceptuje Twój plan."""
        )

#CRITIC AGENT
class CriticAgent(ConversableAgent):
    """Agent oceniający plan i dbający o jego jakość."""
    def __init__(self, llm_config):
        super().__init__(
            name="CriticAgent",
            llm_config=llm_config,
            system_message="""Jesteś 'Recenzentem Jakości'. Twoim zadaniem jest konstruktywna krytyka planu od PlannerAgenta. Oceń go pod kątem praktyczności, realizmu i efektywności.
Twoje Złote Zasady:
1.  **PROSTOTA JEST KLUCZEM:** Agresywnie kwestionuj nadmiernie skomplikowane kroki. Czy naprawdę potrzebujemy KNNImputer, gdy prosta mediana wystarczy?
2.  **JEDNA ZMIANA NA RAZ:** Jeśli plan proponuje stworzenie kilku złożonych cech w jednym kroku, odrzuć to. Zarekomenduj podzielenie tego na osobne, łatwiejsze do weryfikacji kroki. Plan musi być odporny na błędy.
3.  **KONKRETNE SUGESTIE:** Zawsze podawaj konkretną alternatywę. Zamiast 'To jest złe', napisz 'Krok X jest nieoptymalny. Sugeruję Y, ponieważ Z.'

**PROCES ZATWIERDZANIA (KRYTYCZNIE WAŻNE):**
- Jeśli plan wymaga jakichkolwiek poprawek, jasno je opisz i odeślij do PlannerAgenta. **NIE UŻYWAJ** poniższych fraz kluczowych.
- Jeśli plan jest **doskonały** i nie wymaga żadnych zmian, Twoja odpowiedź **MUSI** mieć następującą, ścisłą strukturę:
Najpierw napisz linię:
`OSTATECZNY PLAN:`
Poniżej wklej **CAŁY, KOMPLETNY** plan od PlannerAgenta.
Na samym końcu wiadomości dodaj frazę:
`PLAN_AKCEPTOWANY_PRZEJSCIE_DO_IMPLEMENTACJI`"""
        )


# In[10]:


# --- Konfiguracja czatu grupowego ---
main_agent_configuration={"cache_seed": 42,"seed": 42,"temperature": 0.0,
                        "config_list": basic_config_agent(agent_name=MAIN_AGENT, api_type=API_TYPE_GEMINI, location=LOCATION, project_id=PROJECT_ID)}
critic_agent_configuration ={"cache_seed": 42,"seed": 42,"temperature": 0.0,
                        "config_list": basic_config_agent(api_key=ANTHROPIC_API_KEY,agent_name=CRITIC_MODEL, api_type=API_TYPE_SONNET)}


#---WYWOŁANIE AGENTÓW
trigger_agent = TriggerAgent(llm_config=main_agent_configuration)
planner_agent = PlannerAgent(llm_config=main_agent_configuration)
critic_agent = CriticAgent(llm_config=main_agent_configuration)


# In[11]:


#FUNKCJA CHATU GRUPOWEGO-WYMYŚLANIE PLANU
def run_autogen_planning_phase(input_path: str,inspiration_prompt: str = "") -> Optional[str]:
    """
    Uruchamia fazę planowania z agentami AutoGen i zwraca finalny plan.
    """
    print("\n" + "="*80)
    print("### ### FAZA 1: URUCHAMIANIE PLANOWANIA STRATEGICZNEGO (AutoGen) ### ###")
    print("="*80 + "\n")

    try:
        df_summary = pd.read_csv(input_path, nrows=5)
        data_preview = f"Oto podgląd danych:\n\nKolumny:\n{df_summary.columns.tolist()}\n\nPierwsze 5 wierszy:\n{df_summary.to_string()}"
        
        if inspiration_prompt:
            print("INFO: Dołączam inspiracje z pamięci do fazy planowania.")
            data_preview += "\n\n" + inspiration_prompt
        
    except Exception as e:
        logging.error(f"Nie można wczytać pliku wejściowego {input_path}: {e}")
        return None
    
    
    
    user_proxy = autogen.UserProxyAgent(
       name="UserProxy",
       human_input_mode="NEVER",
       max_consecutive_auto_reply=10,
       is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
       code_execution_config=False,
       system_message="Zarządzasz procesem. Przekaż podgląd danych do TriggerAgenta, a następnie moderuj dyskusję między Plannerem a Krytykiem. Jeśli w wiadomości są inspiracje z przeszłości, przekaż je Plannerowi."
    )

    def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
        messages = groupchat.messages

        # Warunek początkowy, pierwszy mówi TriggerAgent
        if len(messages) <= 1:
            return trigger_agent

        # Standardowy przepływ: Trigger -> Planner -> Critic -> Planner ...
        elif last_speaker is trigger_agent:
            return planner_agent
        elif last_speaker is planner_agent:
            return critic_agent
        elif last_speaker is critic_agent:

            if "PLAN_AKCEPTOWANY_PRZEJSCIE_DO_IMPLEMENTACJI" in messages[-1]['content']:
                return None # To elegancko kończy rozmowę
            else:
                # Jeśli nie, wracamy do Plannera z uwagami
                return planner_agent
        else:
            # Sytuacja awaryjna lub koniec, nie wybieraj nikogo
            return None

    groupchat = autogen.GroupChat(
        agents=[user_proxy, trigger_agent, planner_agent, critic_agent],
        messages=[],
        max_round=15,
        speaker_selection_method=custom_speaker_selection_func
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=main_agent_configuration)

    user_proxy.initiate_chat(manager, message=data_preview)

    # Ekstrakcja finalnego planu
    final_plan = None
    critic_messages = [msg['content'] for msg in groupchat.messages if msg['name'] == 'CriticAgent']
    for msg in reversed(critic_messages):
        if "PLAN_AKCEPTOWANY_PRZEJSCIE_DO_IMPLEMENTACJI" in msg:
            match = re.search(r"OSTATECZNY PLAN:(.*)PLAN_AKCEPTOWANY_PRZEJSCIE_DO_IMPLEMENTACJI", msg, re.DOTALL)
            if match:
                final_plan = match.group(1).strip()
                print("Faza planowania zakończona. Ostateczny plan został zaakceptowany.")
                break
    
    if not final_plan:
        print(" Faza planowania zakończona bez akceptacji planu lub z powodu TERMINATE.")

    
    full_conversation_log = "\n\n".join([f"--- Komunikat od: {msg['name']} ---\n{msg['content']}" for msg in groupchat.messages])

    
    return final_plan, full_conversation_log


# ################################################################################
# ### ### FAZA 2: WYKONANIE PLANU (LangGraph)
# ################################################################################

# In[12]:


#Zasady tworzenia architektury dla wykonawców kodu
class ArchitecturalRule(TypedDict):
    id: str; description: str; check: Callable[[str], bool]; error_message: str

ARCHITECTURAL_RULES: List[ArchitecturalRule] = [
    {"id": "NO_MAIN_BLOCK", "description": "Żadnego bloku `if __name__ == '__main__':`.", "check": lambda code: bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', code)), "error_message": "Wykryto niedozwolony blok `if __name__ == '__main__':`."},
    {"id": "NO_ARGPARSE", "description": "Żadnego `argparse` ani `sys.argv`.", "check": lambda code: bool(re.search(r'import\s+argparse', code)), "error_message": "Wykryto niedozwolony import modułu `argparse`."},
    {"id": "SINGLE_FUNCTION_LOGIC", "description": "Cała logika musi być w funkcji `process_data(input_path: str, output_path: str)`.", "check": lambda code: "def process_data(input_path: str, output_path: str)" not in code, "error_message": "Brak wymaganej definicji funkcji `process_data(input_path: str, output_path: str)`."},
    {"id": "ENDS_WITH_CALL", "description": "Skrypt musi kończyć się **dokładnie jedną linią** w formacie: `process_data(input_path, output_path)  # noqa: F821`. Komentarz `# noqa: F821` jest **obowiązkowy**.", "check": lambda code: not re.search(r'^\s*process_data\(input_path,\s*output_path\)\s*#\s*noqa:\s*F821\s*$', [line for line in code.strip().split('\n') if line.strip()][-1]), "error_message": "Skrypt nie kończy się wymaganym wywołaniem `process_data(input_path, output_path)  # noqa: F821`."},
]

class ArchitecturalRulesManager:
    @staticmethod
    def get_rules_as_string() -> str:
        rules_text = "\n".join(f"        - {rule['description']}" for rule in ARCHITECTURAL_RULES)
        return f"<ARCHITECTURAL_RULES>\n    **Krytyczne Wymagania Dotyczące Struktury Kodu:**\n{rules_text}\n</ARCHITECTURAL_RULES>"


# In[13]:


# --- PROMPTY DLA AGENTÓW LANGCHAIN ---


# In[15]:


class PromptTemplates:
    @staticmethod
    def code_generator(plan: str, available_columns: List[str]) -> str:
        return f"""**Persona:** Ekspert Inżynierii Danych.\n**Plan Biznesowy:**\n{plan}\n
        **Dostępne Kolumny:**\n{available_columns}\n{ArchitecturalRulesManager.get_rules_as_string()}\n
        **Zadanie:** Napisz kompletny skrypt Pythona realizujący plan, przestrzegając wszystkich zasad. Odpowiedź musi zawierać tylko i wyłącznie blok kodu ```python ... ```."""
    
    @staticmethod
    def tool_based_debugger() -> str:
        return """Jesteś 'Głównym Inżynierem Jakości Kodu'. Twoim zadaniem jest nie tylko naprawienie zgłoszonego błędu, ale zapewnienie, że kod będzie działał poprawnie.
- Jeśli błąd to `ModuleNotFoundError`, użyj `request_package_installation`.
- Jeśli błąd to `ImportError` wskazujący na konflikt wersji, również użyj `request_package_installation`, aby zasugerować aktualizację pakietu, który jest źródłem błędu.
- Dla wszystkich innych błędów w kodzie (np. `SyntaxError`, `KeyError`), użyj `propose_code_fix` a następnie przeanalizuj poniższy błąd i wadliwy kod. Twoja praca składa się z dwóch kroków:
1.  **Analiza i Naprawa:** Zidentyfikuj przyczynę błędu i stwórz kompletną, poprawioną wersję całego skryptu.
2.  **Wywołanie Narzędzia:** Wywołaj narzędzie `propose_code_fix`, podając **OBOWIĄZKOWO** dwa argumenty: `analysis` (twoja analiza) oraz `corrected_code` (pełny, naprawiony kod).
Przeanalizuj poniższy błąd i wadliwy kod. """

    @staticmethod
    def create_reporting_prompt(plan: str, original_summary: str, processed_summary: str) -> str:
        return  f"""
**Persona:** Jesteś autonomicznym, starszym Analitykiem Danych. Twoim zadaniem nie jest tworzenie fragmentów kodu, ale dostarczenie kompletnego, gotowego do wdrożenia skryptu w Pythonie, który generuje profesjonalny raport w formacie HTML. Twoja praca musi być w pełni samowystarczalna.

---
## 1. Dostępne Zasoby w Środowisku Wykonawczym

Twój skrypt będzie wykonany w środowisku, w którym następujące zmienne są już zdefiniowane i gotowe do użycia:

- `df_original`: Ramka danych Pandas z danymi *przed* przetwarzaniem.
- `df_processed`: Ramka danych Pandas z danymi *po* przetworzeniu.
- `report_output_path`: String zawierający ścieżkę, pod którą należy zapisać finalny plik HTML (np. 'reports/final_report.html').

---
## 2. Kontekst Biznesowy i Dane

Dane zostały przetworzone zgodnie z następującym planem: {plan}.

Oto podsumowania statystyczne danych, które masz przeanalizować:
{original_summary}
{processed_summary}

---
## 3. Twoje Główne Zadanie: Stworzenie Kompletnego Skryptu Raportującego

Napisz **jeden, kompletny i wykonywalny skrypt w Pythonie**, który realizuje następujące kroki:

1.  **Analiza i Podsumowanie (w HTML):** Stwórz zwięzłe, ale wnikliwe podsumowanie kluczowych zmian między `df_original` a `df_processed`. Zapisz je w zmiennej `summary_html`.
2.  **Generowanie Wizualizacji:** Stwórz co najmniej dwie wartościowe wizualizacje porównawcze (np. histogramy, boxploty) za pomocą Matplotlib, aby zilustrować najważniejsze zmiany (np. wpływ usunięcia wartości odstających na rozkład).
3.  **Konwersja Wykresów:** Każdy wygenerowany wykres musi zostać przekonwertowany do formatu base64 i osadzony w tagu `<img>`.
4.  **Złożenie Raportu HTML:** Skonstruuj kompletny dokument HTML, zawierający zarówno analizę tekstową, jak i osadzone wizualizacje.
5.  **Zapis do Pliku:** Zapisz finalny string HTML do pliku, korzystając ze zmiennej `report_output_path`.

---
## 4. Wymagana Struktura Skryptu (Szablon)

Twój kod musi idealnie pasować do poniższej struktury. Nie modyfikuj jej, jedynie uzupełnij oznaczone sekcje.

```python
# ===================================================================
# === AUTONOMICZNY SKRYPT GENERUJĄCY RAPORT ANALITYCZNY ===
# ===================================================================
# Importy i funkcje pomocnicze są już zapewnione w środowisku,
# ale dla przejrzystości zostaną tu zdefiniowane.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64

# --- Funkcja pomocnicza do osadzania wykresów ---
def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    base64_str = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return base64_str

# --- Zmienne wejściowe (dostępne globalnie w skrypcie) ---
# df_original: pd.DataFrame
# df_processed: pd.DataFrame
# report_output_path: str

# Inicjalizacja listy na wykresy
figures_html_list = []

# ===================================================================
# ### KROK 1: Analiza tekstowa i podsumowanie w HTML ###
# ===================================================================

# <<< UZUPEŁNIJ TĘ SEKCJĘ >>>
# Porównaj kluczowe statystyki, zmiany w rozkładach, liczbę kolumn itp.
# Wynik zapisz w zmiennej summary_html.
summary_html = f\"\"\"
<h2>Podsumowanie Zmian w Danych</h2>
<p>Analiza porównawcza wykazała następujące kluczowe różnice:</p>
<ul>
    <li><strong>Struktura danych:</strong> Liczba kolumn zmieniła się z {len(df_original.columns)} na {len(df_processed.columns)}.</li>
    <li><strong>Wartości odstające:</strong> Maksymalna wartość w kolumnie 'Transaction_Amount' została zredukowana z {df_original['Transaction_Amount'].max():.2f} do {df_processed['Transaction_Amount'].max():.2f}, co świadczy o skutecznej obsłudze outlierów.</li>
    # Dodaj więcej wnikliwych obserwacji...
</ul>
\"\"\"


# ===================================================================
# ### KROK 2: Generowanie wizualizacji porównawczych ###
# ===================================================================

# <<< UZUPEŁNIJ TĘ SEKCJĘ >>>
# Stwórz co najmniej dwa wykresy. Pamiętaj o tytułach i etykietach.

# --- Wykres 1: Porównanie rozkładu kwoty transakcji ---
fig1, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_original['Transaction_Amount'], bins=50, alpha=0.6, label='Oryginalne', color='blue')
ax.hist(df_processed['Transaction_Amount'], bins=50, alpha=0.8, label='Przetworzone', color='green')
ax.set_title('Porównanie Rozkładu Kwoty Transakcji', fontweight='bold')
ax.set_xlabel('Kwota Transakcji')
ax.set_ylabel('Częstość')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
figures_html_list.append(f"<h3>Wykres 1: Dystrybucja Kwot Transakcji</h3>{fig_to_base64(fig1)}")

# --- Wykres 2: (Dodaj kolejny, np. boxplot dla innej zmiennej) ---
# ...


# ===================================================================
# ### KROK 3: Złożenie i zapis finalnego raportu HTML ###
# ===================================================================

# Połącz wszystkie części w jeden dokument HTML
all_figures_html = "".join(figures_html_list)

full_html_report = f\"\"\"
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Raport z Analizy Przetwarzania Danych</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; background-color: #f9f9f9; }}
        .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 2em; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;}}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 4px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Raport Porównawczy Danych</h1>
        {summary_html}
        {all_figures_html}
    </div>
</body>
</html>
\"\"\"

# Zapis do pliku z obsługą błędów
try:
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(full_html_report)
    print(f"Raport został pomyślnie wygenerowany i zapisany jako '{report_output_path}'.")
except IOError as e:
    print(f"Wystąpił błąd podczas zapisywania pliku raportu: {e}")
"""
    def create_meta_auditor_prompt(source_code: str, autogen_conversation: str, langgraph_log: str, final_code: str, final_report: str) -> str:
        return f"""**Persona:** Główny Audytor Systemów AI. Twoim zadaniem jest krytyczna ocena całego procesu AI.
**Dostępne Dane do Analizy:**
1. KOD ŹRÓDŁOWY SYSTEMU:\n```python\n{source_code}\n```
2. ZAPIS ROZMOWY (PLANOWANIE):\n```\n{autogen_conversation}\n```
3. LOGI (WYKONANIE):\n```\n{langgraph_log}\n```
4. FINALNY KOD:\n```python\n{final_code}\n```
5. FINALNY RAPORT (fragment):\n```html\n{final_report[:2000]}\n```
**Zadania Audytorskie (odpowiedz na każde pytanie):**
1. **Ocena Planowania:** Czy dyskusja Planner-Krytyk była efektywna? Czy Krytyk był rygorystyczny?
2. **Ocena Wykonania:** Czy były pętle naprawcze? Jak skuteczny był debugger?
3. **Ocena Produktu:** Czy raport HTML jest użyteczny?
4. **Ocena Promptów Agentów (Analiza Meta):**
    - Na podstawie analizy logów i kodu źródłowego, oceń jakość i precyzję promptów dla poszczególnych agentów (Planner, Krytyk, Debugger, Generator Raportu).
    - Czy któryś z zaobserwowanych problemów (nawet tych naprawionych) mógł wynikać z niejasności w prompcie?
    - Czy widzisz możliwość ulepszenia któregoś z promptów, aby system działał bardziej niezawodnie lub efektywnie w przyszłości?
5. **Rekomendacje do Samodoskonalenia:** Zaproponuj 1-3 konkretne zmiany w kodzie lub promptach, które usprawnią system.
**Format Wyjściowy:** Zwięzły raport tekstowy."""


# In[16]:


#Funkcje pomocnicze, narzędzia dla agentó langchain

def extract_python_code(response: str) -> str:
    response = response.strip()
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match: return match.group(1).strip()
    if response.startswith("'''") and response.endswith("'''"): return response[3:-3].strip()
    if response.startswith('"""') and response.endswith('"""'): return response[3:-3].strip()
    return response

#--funkcja dla pamieci--
def intelligent_truncate(text: str, max_len: int) -> str:
    """Skraca tekst, zachowując jego początek i koniec."""
    if not isinstance(text, str) or len(text) <= max_len:
        return text
    half_len = (max_len - 25) // 2
    start = text[:half_len]
    end = text[-half_len:]
    return f"{start}\n\n[... treść skrócona ...]\n\n{end}"


#Dla inteligentnego debbugera:
class DebugReport(BaseModel):
    analysis: str = Field(description="Techniczna analiza błędu.")
    corrected_code: str = Field(description="Kompletny, poprawiony kod.")

    
class GeneratedPythonScript(BaseModel):
    """
    Model przechowujący kompletny i gotowy do wykonania skrypt w Pythonie.
    """
    script_code: str = Field(description="Kompletny kod w Pythonie, gotowy do bezpośredniego wykonania. Musi zawierać wszystkie niezbędne elementy, takie jak definicje, logikę i zapis pliku.")    
    

class CodeFixArgs(BaseModel):
    analysis: str = Field(description="Techniczna analiza przyczyny błędu i wprowadzonej poprawki w kodzie.")
    corrected_code: str = Field(description="Pełny, kompletny i POPRAWIONY skrypt w Pythonie. Musi być gotowy do wykonania.")
    
class PackageInstallArgs(BaseModel):
    package_name: str = Field(description="Nazwa pakietu, który należy zainstalować, aby rozwiązać błąd 'ModuleNotFoundError'. Np. 'scikit-learn', 'seaborn'.")
    analysis: str = Field(description="Krótka analiza potwierdzająca, że przyczyną błędu jest brakujący pakiet.")

@tool(args_schema=CodeFixArgs)
def propose_code_fix(analysis: str, corrected_code: str) -> None:
    """Użyj tego narzędzia, aby zaproponować poprawioną wersję kodu w odpowiedzi na błąd składniowy lub logiczny."""
    pass

@tool(args_schema=PackageInstallArgs)
def request_package_installation(package_name: str, analysis: str) -> None:
    """Użyj tego narzędzia, aby poprosić o instalację brakującej biblioteki, gdy napotkasz błąd 'ModuleNotFoundError'."""
    pass 

    
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


# In[17]:


#Zmienne przekazywane do grafu LangChian
class AgentWorkflowState(TypedDict):
    plan: str; input_path: str; output_path: str; report_output_path: str
    available_columns: List[str]; generated_code: str; generated_report_code: str
    correction_attempts: int; error_message: Optional[str]; failing_node: Optional[str]
    error_context_code: Optional[str]; debugger_analysis: Optional[str]
    package_to_install: Optional[str]; user_approval_status: Optional[str]
    tool_choice: Optional[str]; tool_args: Optional[Dict]
    source_code: str
    autogen_log: str
    langgraph_log: str
    # --- Pola pamięci ---
    run_id: str
    dataset_signature: str
    error_record_id: Optional[str]
    memory_client: MemoryBankClient
    pending_fix_session: Optional[Dict[str, Any]] 
    


# In[18]:


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
    
    llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    prompt = PromptTemplates.code_generator(state['plan'], state['available_columns'])
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
    
    # llm = ChatAnthropic(model_name=CODE_MODEL, temperature=0.0, max_tokens=2048)
    llm = ChatVertexAI(model_name=MAIN_AGENT,temperature=0.0, project=PROJECT_ID, location=LOCATION)
    tools = [propose_code_fix, request_package_installation]
    llm_with_tools = llm.bind_tools(tools)
    prompt = PromptTemplates.tool_based_debugger()
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
        # ... (cała logika generowania raportu audytora, tak jak w oryginale)
        # Załóżmy, że wynikiem jest zmienna 'audit_report'
        final_report_content = "Brak raportu do analizy."
        try:
            with open(state['report_output_path'], 'r', encoding='utf-8') as f:
                final_report_content = f.read()
        except Exception: pass
        
        llm = ChatAnthropic(model_name=CRITIC_MODEL, temperature=0.0, max_tokens=2048)
        prompt = PromptTemplates.create_meta_auditor_prompt(
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


# ################################################################################
# ### ### Główny blok uruchomieniowy
# ################################################################################

# In[19]:


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    system_source_code = read_source_code("Agents_beta.ipynb") # Pamiętaj o poprawnej nazwie pliku

    # --- Inicjalizacja Pamięci i Uruchomienia ---
    memory_client = MemoryBankClient(client=client, agent_engine=agent_engine)
    run_id = str(uuid.uuid4())
    
    print("\n--- ODPYTYWANIE PAMIĘCI O INSPIRACJE ---")
    inspiration_prompt = ""
    dataset_signature = ""
    try:
        df_preview = pd.read_csv(INPUT_FILE_PATH, nrows=0)
        dataset_signature = memory_client.create_dataset_signature(df_preview)
        past_memories = memory_client.query_memory(
            query_text="Najlepsze strategie i kluczowe wnioski dotyczące przetwarzania danych",
            scope={"dataset_signature": dataset_signature},
            top_k=3
        )
        if past_memories:
            inspirations = []
            for mem in past_memories:
                if mem.memory_type == MemoryType.SUCCESSFUL_PLAN and 'key_insight' in mem.content:
                    inspirations.append(f"SPRAWDZONY WNIOSEK Z PLANU: {mem.content['key_insight']}")
                elif mem.memory_type == MemoryType.SUCCESSFUL_FIX and 'key_takeaway' in mem.content:
                    inspirations.append(f"NAUCZKA Z NAPRAWIONEGO BŁĘDU: {mem.content['key_takeaway']}")
            if inspirations:
                inspiration_prompt = "--- INSPIRACJE Z POPRZEDNICH URUCHOMIEŃ ---\n" + "\n".join(inspirations)
                print("INFO: Pomyślnie pobrano inspiracje z pamięci.")
        else:
            print("INFO: Nie znaleziono inspiracji w pamięci dla tego typu danych.")
    except Exception as e:
        print(f"OSTRZEŻENIE: Nie udało się pobrać inspiracji z pamięci: {e}")

    # --- Krok 1: Faza planowania AutoGen ---
    final_plan, autogen_log = run_autogen_planning_phase(input_path=INPUT_FILE_PATH, inspiration_prompt=inspiration_prompt)

    # Zapis logu z planowania (zawsze)
    save_autogen_conversation_log(log_content=autogen_log, file_path="reports/autogen_planning_conversation.log")

    # --- Krok 2: Faza wykonania LangGraph ---
    if final_plan:
        print("\n" + "="*80)
        print("### ### FAZA 2: URUCHAMIANIE WYKONANIA PLANU (LangGraph) ### ###")
        print("="*80 + "\n")
        
        workflow = StateGraph(AgentWorkflowState)
        
        # ZMIANA: Dodajemy nowy węzeł commit_memory_node do listy
        nodes = [
            "schema_reader", "code_generator", "architectural_validator", 
            "data_code_executor", "universal_debugger", "apply_code_fix", 
            "human_approval", "package_installer", "reporting_agent", 
            "report_executor", "human_escalation", "sync_report_code",
            "commit_memory" # NOWY WĘZEŁ
        ]
        for name in nodes: workflow.add_node(name, globals()[f"{name}_node"])

        # --- Definicja Krawędzi Grafu ---
        workflow.set_entry_point("schema_reader")
        workflow.add_edge("schema_reader", "code_generator")
        workflow.add_edge("code_generator", "architectural_validator")

        # Funkcja routująca, której możemy używać wielokrotnie
        def should_continue_or_debug(state: AgentWorkflowState) -> str:
            """Sprawdza, czy w stanie jest błąd i decyduje o dalszej ścieżce."""
            if state.get("error_message"):
                if state.get("correction_attempts", 0) >= MAX_CORRECTION_ATTEMPTS:
                    return "request_human_help"
                return "call_debugger"
            # Jeśli nie ma błędu, kontynuuj normalną ścieżkę
            return "continue"

        # 1. KRAWĘDŹ WARUNKOWA po walidatorze architektury (KLUCZOWA ZMIANA)
        workflow.add_conditional_edges(
            "architectural_validator",
            should_continue_or_debug,
            {
                "call_debugger": "universal_debugger",
                "request_human_help": "human_escalation",
                "continue": "data_code_executor" # Przejdź dalej tylko jeśli jest OK
            }
        )

        # 2. KRAWĘDŹ WARUNKOWA po wykonaniu kodu danych
        workflow.add_conditional_edges(
            "data_code_executor",
            should_continue_or_debug,
            {
                "call_debugger": "universal_debugger",
                "request_human_help": "human_escalation",
                "continue": "commit_memory" # Jeśli sukces, idź do zapisu w pamięci, a NIE do END
            }
        )

        # Ścieżka sukcesu i pozostałe krawędzie
        workflow.add_edge("commit_memory", "reporting_agent")
        workflow.add_edge("reporting_agent", "report_executor")

        # Krawędź warunkowa po wykonaniu raportu
        workflow.add_conditional_edges(
            "report_executor",
            should_continue_or_debug,
            {
                "call_debugger": "universal_debugger",
                "request_human_help": "human_escalation",
                "continue": END # Dopiero tutaj kończymy pracę po sukcesie
            }
        )

        # Ścieżki naprawcze i eskalacji (bez zmian)
        workflow.add_edge("human_escalation", END)
        workflow.add_edge("package_installer", "data_code_executor") # Wracamy do wykonania po instalacji

        def route_after_fix(state):
            failing_node = state.get("failing_node")
            if failing_node == "report_executor":
                return "sync_report_code"
            # Po każdej innej naprawie wracamy do walidacji architektonicznej
            return "architectural_validator"

        workflow.add_edge("sync_report_code", "report_executor")
        workflow.add_conditional_edges("apply_code_fix", route_after_fix)

        def route_from_debugger(state):
            if state.get("tool_choice") == "propose_code_fix":
                return "apply_code_fix"
            if state.get("tool_choice") == "request_package_installation":
                return "human_approval"
            return "human_escalation"

        workflow.add_conditional_edges("universal_debugger", route_from_debugger)
        workflow.add_conditional_edges("human_approval", lambda s: s.get("user_approval_status"), {
            "APPROVED": "package_installer",
            "REJECTED": "universal_debugger"
        })

        app = workflow.compile()
        
        initial_state = {
            "plan": final_plan, 
            "input_path": INPUT_FILE_PATH,
            "output_path": "reports/processed_data.csv",
            "report_output_path": "reports/transformation_report.html",
            "correction_attempts": 0, 
            "source_code": system_source_code,
            "autogen_log": autogen_log,
            "memory_client": memory_client,
            "run_id": run_id,
            "dataset_signature": dataset_signature,
            "pending_fix_session": None # ZMIANA: Dodanie nowego pola do stanu początkowego
        }
        
        # --- Uruchomienie grafu z przechwytywaniem logów ---
        langgraph_log = ""
        final_run_state = initial_state.copy()
        
        for event in app.stream(initial_state, {"recursion_limit": 50}):
            for node_name, state_update in event.items():
                if "__end__" not in node_name:
                    print(f"--- Krok: '{node_name}' ---")
                    if state_update: # Zabezpieczenie przed błędem 'NoneType'
                        printable_update = state_update.copy()
                        for key in ["generated_code", "corrected_code", "generated_report_code", "error_context_code"]:
                            if key in printable_update and printable_update[key]:
                                print(f"--- {key.upper()} ---")
                                print(printable_update[key])
                                print("-" * (len(key) + 8))
                                del printable_update[key]
                        if printable_update:
                            print(json.dumps(printable_update, indent=2, default=str))
                        
                        log_line = f"--- Krok: '{node_name}' ---\n{json.dumps(state_update, indent=2, default=str)}\n"
                        langgraph_log += log_line
                        final_run_state.update(state_update)
                    else:
                        print("  [INFO] Węzeł zakończył pracę bez aktualizacji stanu.")
                    print("-" * 20 + "\n")

        # Zapis logu z wykonania (po zakończeniu pętli)
        save_langgraph_execution_log(log_content=langgraph_log, file_path="reports/langgraph_execution.log")

        # Uruchomienie audytora
        final_run_state['langgraph_log'] = langgraph_log
        meta_auditor_node(final_run_state)

        print("\n\n--- ZAKOŃCZONO PRACĘ GRAFU I AUDYT ---")
    else:
        print("Proces zakończony. Brak planu do wykonania.")


# In[ ]:





# In[ ]:




