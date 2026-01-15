import json
import os
import re
from typing import Any, Dict
from litellm import completion
from dotenv import load_dotenv

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

# Load the environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

def clean_json_response(content: str) -> str:
    """
    Remove the MD formatting from the JSON response.
    """
    content = content.strip()
    
    # Remove the leading ```.
    content = re.sub(r'^```json\s*\n?', '', content)
    content = re.sub(r'^```\s*\n?', '', content)
    
    # Remove the trailing ```.
    content = re.sub(r'\n?```\s*$', '', content)
    
    return content.strip()

def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.

    # Create the prompt

    system_prompt = """You are a travel expert. Generate a detailed travel itinerary in JSON format.
    Return ONLY valid JSON with the following structure:
    {
      "destination": "destination name",
      "price_range": "budget/moderate/luxury",
      "ideal_visit_times": ["season1", "season2"],
      "top_attractions": [
        {"name": "attraction1", "description": "brief description"},
        {"name": "attraction2", "description": "brief description"}
      ]
    }"""

    user_prompt = f"Create a travel itinerary for {destination}"

    try:
        # Call the LiteLLM completion API
        response = completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"} 
        )

        # Extract the response content
        content = response.choices[0].message.content
        cleaned_content = clean_json_response(content)

        # Parse JSON from the response
        data = json.loads(cleaned_content)

        # Validate required fields
        required_fields = ["destination", "price_range",
                           "ideal_visit_times", "top_attractions"]
        if not all(field in data for field in required_fields):
            raise ValueError("Response missing required fields")

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {e}")
    except Exception as e:
        raise Exception(f"Error calling LiteLLM API: {e}")
