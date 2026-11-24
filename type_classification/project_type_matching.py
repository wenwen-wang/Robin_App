from mistralai import Mistral
import os


def _get_mistral_api_key():
    """
    Get API key from Streamlit secrets or environment variable.
    """
    # 1) Try Streamlit secrets (when running streamlit)
    try:
        import streamlit as st
        key = st.secrets.get("MISTRAL_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # 2) Fallback to environment variable
    key = os.getenv("MISTRAL_API_KEY")
    if key:
        return key

    return None


def match_project_type(project_proposal):

    ProjectType = [
        'Community_cargo_bike',
        "Concept_development_workshop",
        "Corporate_social_engagement_platform",
        "Free_shop",
        "Idea_consulting",
        "Mobile_housing",
        "Neighbourhood_bazaar",
        "Neighbourhood_governance",
        "Open_makerspace",
        "Parklet",
        "Public_bookcase",
        "Urban_green_spaces",
        "Water_sensitive_urban_intervention"]

    system_prompt = "You are an project classifier, give a project description, please classify the project into three types it belongs to with the confidence score." \
                    " The possible values of project types are " \
                    f"Possible Project_Type values: {ProjectType}, only retruen the project type with the highest confidence score in string, no markdown, no explaination."
    user_prompt = f"Project description is: {project_proposal}."

    MISTRAL_API_KEY = _get_mistral_api_key()
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model="mistral-small-latest",
        temperature=0.0,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    project_type = response.choices[0].message.content
    return project_type

