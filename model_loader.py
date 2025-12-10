from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "yashpwr/resume-ner-bert-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def extract_entities(text):
    """
    Returns entities with normalized entity_group names
    """
    raw_entities = ner_pipeline(text)
    entities = []
    for e in raw_entities:
        group = e["entity_group"].lower()
        # Map possible variants to standard keys
        if "skill" in group:
            group = "skills"
        elif "degree" in group:
            group = "degree"
        elif "experience" in group:
            group = "experience"
        entities.append({
            "entity_group": group,
            "word": e["word"],
            "score": e.get("score", 0)
        })
    return entities
