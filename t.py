import json
for line in open("enriched.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    if rec["hash"] == "92629647083470225439811734491942287035":   # arm arch_get_unmapped_area
        print(rec["change"]["function_diff"][:700])
        print(repr(rec["func"][:150]), rec["location"])