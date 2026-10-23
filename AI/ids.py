
import re
import uuid
USER_NAMESPACE = uuid.UUID("6f1c2a52-8a3e-4d1e-9b7a-2f4c0d9e7b11")


def normalize_name(name):
    return " ".join(name.strip().lower().split())


def normalize_phone(phone):
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("84") and len(digits) == 11:  
        digits = "0" + digits[2:]
    return digits


def create_user_id(name, phone):
    key = f"{normalize_name(name)}:{normalize_phone(phone)}"
    return str(uuid.uuid5(USER_NAMESPACE, key))