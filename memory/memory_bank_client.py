import json
import hashlib
import pandas as pd
from typing import Dict, List, Optional
import vertexai
from vertexai import agent_engines
from memory_models import MemoryRecord


class MemoryBankClient:
    def __init__(self, client: vertexai.Client, agent_engine):
        """
        Inicjalizuje klienta z głównym obiektem klienta Vertex AI 
        oraz z gotowym obiektem agent_engine.
        """
        if not client or not agent_engine:
            raise ValueError("Klient Vertex AI oraz Agent Engine muszą być poprawnie zainicjalizowane.")
        
        self.client = client
        self.agent_engine = agent_engine
        self.engine_name = agent_engine.resource_name
        print(f"INFO: MemoryBankClient gotowy do pracy z silnikiem: {self.engine_name}")
        
    
    
    
    def create_dataset_signature(self, df_preview: pd.DataFrame) -> str:
        """Tworzy unikalny identyfikator dla zbioru danych."""
        s = "".join(df_preview.columns) + str(df_preview.shape)
        return hashlib.md5(s.encode()).hexdigest()

    def add_memory(self, record: MemoryRecord):
        """Zapisuje ustrukturyzowane wspomnienie w Agent Engine."""
        try:
            fact_to_remember = record.model_dump_json()
            scope = {"dataset_signature": record.dataset_signature}
            
            
            self.client.agent_engines.create_memory(
                name=self.engine_name,
                fact=fact_to_remember, 
                scope=scope
            )
            print(f"INFO: Zapisano wspomnienie typu '{record.memory_type}' w zakresie {scope}")
        except Exception as e:
            print(f"BŁĄD ZAPISU PAMIĘCI: {e}")

#     def query_memory(self, query_text: str, scope: Dict, top_k: int = 5) -> List[MemoryRecord]:
#         """
#         Odpytuje pamięć semantycznie i zwraca listę ustrukturyzowanych wspomnień.
#         """
#         retrieved_mems = []
#         try:
#             print(f"INFO: Odpytuję pamięć semantycznie z zapytaniem '{query_text}' w zakresie {scope}")

#             # Tworzymy słownik z parametrami wyszukiwania, zgodnie z Twoim znaleziskiem
#             search_params = {
#                 "search_query": query_text,
#                 "top_k": top_k
#             }

#             # Wywołujemy API z poprawnym argumentem: similarity_search_params
#             memories_iterator = self.client.agent_engines.retrieve_memories(
#                 name=self.engine_name,
#                 scope=scope,
#                 similarity_search_params=search_params
#             )

#             for mem in memories_iterator:
#                 print("mem", dir(mem))
#                 record = MemoryRecord.model_validate_json(mem.memory)
#                 retrieved_mems.append(record)

#             print(f"INFO: Znaleziono {len(retrieved_mems)} pasujących wspomnień.")
#             return retrieved_mems

#         except Exception as e:
#             print(f"BŁĄD ODCZYTU PAMIĘCI: {e}")
#             return []
        
        
    def query_memory(self, query_text: str, scope: Dict, top_k: int = 5) -> List[MemoryRecord]:
        """
        Odpytuje pamięć semantycznie, poprawnie odczytując zagnieżdżoną treść wspomnienia.
        """
        retrieved_mems = []
        try:
            print(f"INFO: Odpytuję pamięć semantycznie z zapytaniem '{query_text}' w zakresie {scope}")

            search_params = {
                "search_query": query_text,
                "top_k": top_k
            }

            memories_iterator = self.client.agent_engines.retrieve_memories(
                name=self.engine_name,
                scope=scope,
                similarity_search_params=search_params
            )

            for i, mem in enumerate(memories_iterator):
                try:

                    json_string_to_parse = mem.memory.fact

                    record = MemoryRecord.model_validate_json(json_string_to_parse)
                    retrieved_mems.append(record)
                    print("udany plan:", record)
                except Exception as e:
                    print(f"⚠️ OSTRZEŻENIE: Pominięto uszkodzony lub niekompatybilny rekord pamięci (pozycja {i}). Błąd: {e}")
                    continue

            print(f"INFO: Znaleziono i poprawnie przetworzono {len(retrieved_mems)} pasujących wspomnień.")
            return retrieved_mems

        except Exception as e:
            print(f"KRYTYCZNY BŁĄD ODCZYTU PAMIĘCI: Nie udało się wykonać zapytania. Błąd: {e}")
            return []