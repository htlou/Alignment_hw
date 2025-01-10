import datasets
import json

path = "/data/align-anything/hantao/data/PKU-SafeRLHF-single-dimension"

dataset = datasets.load_dataset(path)

# print(dataset['test'][0])

chosen = []
rejected = []

for sample in dataset['test']:
    if sample['better_response_id'] == 1:
        chosen.append({
            'prompt': sample['prompt'],
            'answer': sample['response_1'],
        })
        rejected.append({
            'prompt': sample['prompt'],
            'answer': sample['response_0'],
        })
    else:
        chosen.append({
            'prompt': sample['prompt'],
            'answer': sample['response_0'],
        })
        rejected.append({
            'prompt': sample['prompt'],
            'answer': sample['response_1'],
        })

print(chosen[0])
print(rejected[0])

chosen_path = "/data/align-anything/hantao/Alignment_hw/hw2/data/chosen.json"
rejected_path = "/data/align-anything/hantao/Alignment_hw/hw2/data/rejected.json"

with open(chosen_path, 'w') as f:
    json.dump(chosen, f, indent=4)

with open(rejected_path, 'w') as f:
    json.dump(rejected, f, indent=4)