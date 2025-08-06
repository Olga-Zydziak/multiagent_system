import json

def compile_rules():
    print("--- 1. Wczytywanie reguł z `customer_service_rules.json` ---")
    with open("rules.json", 'r', encoding='utf-8') as f:
        rules_data = json.load(f)

    prolog_code = ""
    for rule in rules_data.get("rules", []):
        prolog_code += (
            f"policy_violation(Content, '{rule['error_message']}') :- "
            f"{rule['prolog_condition']}.\n"
        )
    
    print("--- 2. Kompilowanie reguł do `customer_policies.pl` ---")
    with open("customer_policies.pl", "w", encoding="utf-8") as f:
        preamble = """
% Plik wygenerowany dynamicznie
:- style_check(-singleton).
contains_substring(String, Substring) :- sub_string(String, _, _, _, Substring).
"""
        f.write(preamble.strip() + "\n\n" + prolog_code.strip())
    print("  [SUKCES] Pomyślnie skompilowano reguły.")

if __name__ == "__main__":
    compile_rules()