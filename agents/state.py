from typing import TypedDict, List, Callable, Dict, Optional, Union, Any
from memory.memory_bank_client import MemoryBankClient

#Zmienne przekazywane do grafu LangChian
class AgentWorkflowState(TypedDict):
    config: Dict[str, Any]
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