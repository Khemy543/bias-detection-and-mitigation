import hashlib

obj = {
    "showTaxPrice": "VATExcluded",
    "userPriceList": 1,
    "userCache": "2b44928ae11fb9384c4cf38708677c48",
    "route": "/producten/planken"
}

# Sort keys, take values, join with delimiter
values_string = "|".join(str(obj[k]) for k in sorted(obj.keys()))

hash_value = hashlib.md5(values_string.encode("utf-8")).hexdigest()

print(values_string)
print(hash_value)
