import json 
import gc
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import DistilBertForQuestionAnswering
import numpy as np
import evaluate
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.optim import AdamW
import collections
from tqdm.auto import tqdm
import transformers
from numpy import inf
import random 

class AnalogyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(start_logits, end_logits, features, contexts, true_answers, true_pos):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[int(feature["example_id"])].append(idx)


    predicted_answers = []
    theoretical_answers = []
    n_best = 20
    max_answer_length = 500

    # print(len(start_logits), len(end_logits))
 
    
    for idx,context in enumerate(tqdm(contexts)):
       
        example_id = idx

        answers = []
        updated = False
        best_answer = {"text":'',"logit_score":-inf}
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            # print(feature_index,"1"*20)
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            # print(start_indexes, end_indexes)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # print(offsets[start_index], offsets[end_index])
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        # print("here")
                        continue
                    logit_score = start_logit[start_index] + end_logit[end_index] 
                    if logit_score > best_answer['logit_score']:
                        best_answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": logit_score,
                        }
                        updated = True
                    # print(answer)
                    #answers.append(answer)
                    #print(len(answers))
        # print(len(answers))
        
        # Select the answer with the best score
        if updated:
            #best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": str(example_id), "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": str(example_id), "prediction_text": ""})

        

    theoretical_answers = [{"id": str(idx), "answers": {"text":[ans], "answer_start": [int(true_pos[idx][0])]}} for idx,ans in enumerate(true_answers)]
    #print(predicted_answers, theoretical_answers)
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def read_data(path):
    with open(path, 'r') as f:
        squad_data = f.read()

    contexts = []
    pos = []
    answers = []
    ids = []
    for row in squad_data.split('\n'):
        if row.strip()!= '':
            row_dict = json.loads(row.strip())
            contexts.append(row_dict['webpage'])
            answers.append(row_dict['analogy']) 
            ids.append(row_dict['key'])
            pos.append(row_dict['pos'])
            assert row_dict['analogy'] == row_dict['webpage'][row_dict['pos'][0]:row_dict['pos'][1]]
    
    return contexts, pos, answers, ids

def undersample(inputs, neg_sample_idxs, pos_sample_idxs):

    random.shuffle(neg_sample_idxs)
    sel_neg_sample_idxs = neg_sample_idxs[:len(pos_sample_idxs)]

    sampled_inputs = {}
    
    for k,vs in inputs.items():
        sampled_vs = []
        print("before:",len(vs))
        for idx,v in enumerate(vs):
            if idx in sel_neg_sample_idxs or idx in pos_sample_idxs:
                sampled_vs.append(v)
        sampled_inputs[k] = sampled_vs
        print("after:",len(sampled_vs))
    return sampled_inputs


def preprocess_training_examples(contexts, answers, pos, sample=True):
    max_length = 384
    stride = 128

    if model_name == "longformer":
        max_length = 3072
        stride = 1024

    inputs = tokenizer(
        contexts,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True
    )

    print(len(contexts),len(inputs['input_ids']))

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    neg_sample_idxs = []
    pos_sample_idxs = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        start_char = pos[sample_idx][0]
        end_char = pos[sample_idx][1]
        sequence_ids = inputs.sequence_ids(i)

        # # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 0:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 0:
            idx += 1
        context_end = idx - 1

        # print(context_start, context_end, start_char, end_char, offset[context_start], offset[context_end])

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            neg_sample_idxs.append(i)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
            pos_sample_idxs.append(i)

        # print(start_positions[-1],end_positions[-1])
        labeled_answer = tokenizer.decode(inputs["input_ids"][i][start_positions[-1] : end_positions[-1] + 1])
        #print(labeled_answer)
        #print(answers[sample_idx])
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    if model_name == 'longformer':
        global_attention_mask = torch.zeros_like(torch.Tensor(inputs['input_ids']))
        # global attention on cls token
        global_attention_mask[:, 0] = 1
        inputs['global_attention_mask'] = global_attention_mask

    if sample:
        inputs = undersample(inputs, neg_sample_idxs, pos_sample_idxs)

    return AnalogyDataset(inputs)

def preprocess_validation_examples(contexts):
    max_length = 384
    stride = 128
    if model_name=="longformer":
        max_length = 3072
       
        stride = 1024
    inputs = tokenizer(
        contexts,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True
    )
    print(len(contexts),len(inputs['input_ids']))
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(sample_idx)

    if model_name == 'longformer':
        global_attention_mask = torch.zeros_like(torch.Tensor(inputs['input_ids']))
        # global attention on cls token
        global_attention_mask[:, 0] = 1
        inputs['global_attention_mask'] = global_attention_mask
    inputs["example_id"] = example_ids
    return AnalogyDataset(inputs)

if __name__ == '__main__':
    transformers.logging.set_verbosity_info()
    for i in range(1):

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        #model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-base-4096")
        metric = evaluate.load("squad")

        model_name = 'bert'
        lr = 5e-5

        training_args = TrainingArguments(
        output_dir="model_{}_{}_{}".format(model_name,lr,i+2),
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )

      

        train_contexts,  train_pos, train_answers, train_ids = read_data('../data/split{}/train.jsonl'.format(i+2))
        
        val_contexts, val_pos, val_answers, val_ids = read_data('../data/split{}/test.jsonl'.format(i+2))
       
        train_dataset = preprocess_training_examples(train_contexts, train_answers, train_pos)
        val_dataset = preprocess_validation_examples(val_contexts)

        print(len(train_dataset))
        

        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
        trainer.train(resume_from_checkpoint=True)
        print(len(val_dataset))
        del train_dataset
        del train_contexts
        del train_answers
        del train_pos
        del train_ids
        gc.collect()
        predictions, _, _ = trainer.predict(val_dataset)
        
        start_logits, end_logits = predictions
        print("done predicting")
        print(compute_metrics(start_logits, end_logits, val_dataset, val_contexts, val_answers, val_pos))