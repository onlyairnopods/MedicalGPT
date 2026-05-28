import json

#src = "/cephfs/songyue/zzlai/MedicalGPT/mystuff/selected_huatuo.jsonl"
#dst = "/cephfs/songyue/zzlai/MedicalGPT/mystuff/selected_huatuo_clean.jsonl"
src = "/cephfs/songyue/zzlai/MedicalGPT/mystuff/analysis_out_v3/selected_train_zh_0_share_v3_filtered.jsonl"
dst = "/cephfs/songyue/zzlai/MedicalGPT/mystuff/analysis_out_v3/selected_train_zh_0_share_v3_filtered_clean.jsonl"

num_total = 0
num_kept = 0
num_bad_json = 0
num_bad_sample = 0
num_bad_turn = 0

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        num_total += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            num_bad_json += 1
            continue

        convs = obj.get("conversations")
        if not isinstance(convs, list) or len(convs) == 0:
            num_bad_sample += 1
            continue

        new_convs = []
        valid = True

        for msg in convs:
            if not isinstance(msg, dict):
                valid = False
                break

            role = msg.get("from")
            value = msg.get("value")

            if not isinstance(role, str) or not isinstance(value, str):
                valid = False
                break

            # 关键：重新构造，只保留 from/value
            new_convs.append({
                "from": role,
                "value": value,
            })

        if not valid or len(new_convs) == 0:
            num_bad_turn += 1
            continue

        clean_obj = {
            "id": str(obj.get("id", line_no)),
            "conversations": new_convs,
            # "lang": obj.get("lang", "") if isinstance(obj.get("lang", ""), str) else "",
        }

        fout.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")
        num_kept += 1

print("total     =", num_total)
print("kept      =", num_kept)
print("bad_json  =", num_bad_json)
print("bad_sample=", num_bad_sample)
print("bad_turn  =", num_bad_turn)
print("saved to  =", dst)



# from datasets import load_dataset
# huatuo = load_dataset("json", data_files="/cephfs/songyue/zzlai/MedicalGPT/mydata/sharegpt_zh_38K_clean.jsonl", split="train")
# huatuo = load_dataset("json", data_files="/cephfs/songyue/zzlai/MedicalGPT/mydata/HuatuoGPT2-GPT4-SFT-140K.jsonl", split="train")
