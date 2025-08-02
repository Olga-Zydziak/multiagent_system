from pydantic import BaseModel, Field
from langchain_core.tools import tool
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
    
    
#narzędzia dla langchain agentów    
@tool(args_schema=CodeFixArgs)
def propose_code_fix(analysis: str, corrected_code: str) -> None:
    """Użyj tego narzędzia, aby zaproponować poprawioną wersję kodu w odpowiedzi na błąd składniowy lub logiczny."""
    pass

@tool(args_schema=PackageInstallArgs)
def request_package_installation(package_name: str, analysis: str) -> None:
    """Użyj tego narzędzia, aby poprosić o instalację brakującej biblioteki, gdy napotkasz błąd 'ModuleNotFoundError'."""
    pass 