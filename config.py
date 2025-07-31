def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Pobiera wartość sekretu z Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
   
    return response.payload.data.decode("UTF-8")


class ApiType(Enum):
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    def __str__(self):
        return self.value
    
    
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