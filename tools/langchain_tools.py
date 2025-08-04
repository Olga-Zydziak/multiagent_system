from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .utils import TOOL_REGISTRY
#--BaseModel

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

class ReportSummary(BaseModel):
    """Przechowuje podsumowanie analityczne w formacie HTML."""
    summary_html: str = Field(description="Tekst podsumowania w formacie HTML, zawierający tagi takie jak <h2> i <ul>.")

class PlottingCode(BaseModel):
    """Przechowuje kod Pythona do generowania wizualizacji."""
    code: str = Field(description="Czysty kod w Pythonie do generowania figur matplotlib.")


    
class InspectToolArgs(BaseModel):
    tool_name: str = Field(description="Nazwa funkcji/narzędzia do inspekcji, np. 'embed_plot_to_html'.")
    
#narzędzia dla langchain agentów    
@tool(args_schema=CodeFixArgs)
def propose_code_fix(analysis: str, corrected_code: str) -> None:
    """Użyj tego narzędzia, aby zaproponować poprawioną wersję kodu w odpowiedzi na błąd składniowy lub logiczny."""
    pass

@tool(args_schema=PackageInstallArgs)
def request_package_installation(package_name: str, analysis: str) -> None:
    """Użyj tego narzędzia, aby poprosić o instalację brakującej biblioteki, gdy napotkasz błąd 'ModuleNotFoundError'."""
    pass 


@tool(args_schema=InspectToolArgs)
def inspect_tool_code(tool_name: str) -> str:
    """Użyj tego narzędzia, aby przeczytać kod źródłowy wewnętrznej funkcji systemowej.
    Jest to przydatne, gdy podejrzewasz, że błąd (np. NameError) leży w narzędziu, a nie w kodzie, który analizujesz."""
    if tool_name in TOOL_REGISTRY:
        source_code = inspect.getsource(TOOL_REGISTRY[tool_name])
        return f"Oto kod źródłowy narzędzia '{tool_name}':\n```python\n{source_code}\n```"
    return f"BŁĄD: Nie znaleziono narzędzia o nazwie '{tool_name}'."