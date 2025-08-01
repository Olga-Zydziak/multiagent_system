import pandas as pd
import re
from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
from .autogen_agents import TriggerAgent,PlannerAgent,CriticAgent
import autogen
from autogen import Agent, ConversableAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent





#FUNKCJA CHATU GRUPOWEGO-WYMYŚLANIE PLANU
def run_autogen_planning_phase(input_path: str,trigger_agent: TriggerAgent,planner_agent: PlannerAgent, critic_agent: CriticAgent, manager_agent_config:Dict,inspiration_prompt: str = "") -> Optional[str]:
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
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_agent_config)

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
