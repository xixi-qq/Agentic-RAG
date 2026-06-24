
from apps.rag.schemas import RetrieveItem
from config.agent_config import model

def get_content(retrieve_results):
    total_content = []
    for retrieve_result in retrieve_results:
        source = "filename: " + retrieve_result.metadata.filename + "; page: " + str(retrieve_result.metadata.page_number)
        relevance = str(retrieve_result.score)
        content = retrieve_result.content
        total_content.append(source + "\n" + relevance+ "\n" + content)
    return "\n\n".join(total_content)
