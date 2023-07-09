import json 
import gc
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import DistilBertForQuestionAnswering
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.optim import AdamW
import collections
from tqdm.auto import tqdm
import transformers
from numpy import inf
import random 
import re 
from transformers import DataCollatorForTokenClassification
from transformers import pipeline

from squad_eval import * 


class AnalogyDataset(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		return item

	def __len__(self):
		return len(self.encodings['input_ids'])



def compute_metrics_tok(p):
	label_list = ["O","B-Concept","I-Concept"]
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [
		[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]

	results = seqeval.compute(predictions=true_predictions, references=true_labels)
	return {
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"f1": results["overall_f1"],
		"accuracy": results["overall_accuracy"],
	}

def cncpt_avg(all_scores):
	score_by_idx = {}
	# print(all_scores)
	for scores in all_scores:
		for sc, arg_idx in scores:
			try:
				score_by_idx[arg_idx].append(sc)
			except:
				score_by_idx[arg_idx] = [sc]
	avg_scores = {}	
	for k,vs in score_by_idx.items():
		avg_scores[k] = np.average(np.array(vs))
	return avg_scores

	

def compute_metrics_squad(classifier, samples, ofile):
	f1 = 0
	em = 0
	total = 0
	for idx,s in enumerate(samples['tokens']):
		predictions = classifier(' '.join(s))
		theoretical_answers = [{}]
		pred_answers = {}
		prev_index = -1 
		pred_concepts = []
		concept = ''
		total += 1
		# print('='*20)
		# print("preds", predictions)

		for p in predictions:
			pred_concepts.append(p['word'])

		

		true_concepts = samples['concepts'][idx]

		# print("true",true_concepts)
		# print("pred:",pred_concepts)

		all_f1s = []
		all_ems = []
		for cncpt in pred_concepts:
			# print(cncpt)
			pred_answers[str(idx)] =  cncpt 
			theoretical_answers[0]["paragraphs"] = [{"qas":[{"id": str(idx), "answers": [{"text":c} for c in true_concepts]}]}]

			res = compute_score(theoretical_answers, pred_answers)
			# print("f1s",res['f1'])
			# print("ems",res['exact_match'])
			all_f1s.append(res['f1'])
			all_ems.append(res['exact_match'])

		f1_by_cncpt = cncpt_avg(all_f1s)
		em_by_cncpt = cncpt_avg(all_ems)

		# print("f1 by cncpt",f1_by_cncpt)
		# print("em by concept", em_by_cncpt)

		loc_f1 = 0
		loc_em = 0
		for k,v in f1_by_cncpt.items():
			loc_f1 += v 

		for k,v in em_by_cncpt.items():
			loc_em += v 

		# print("loc",loc_em, loc_f1)

		f1 += loc_f1/max(len(f1_by_cncpt),1)
		em += loc_em/max(len(em_by_cncpt),1)

		# print("f1:",f1)
		# print("em:",em)
		with open(ofile,'a') as f:
			f.write(json.dumps({"f1":f1,"exact_match":em})+'\n')

	f1 = 100.0 * f1/total
	em = 100.0 * em/total

	return {"f1":f1,"exact_match":em}

def form_mapping_dict(txt):
	tokens_map = {}
	widx = 1
	tokens_map[0] = 0

	for cidx,t in enumerate(txt):
		if t == ' ':
			tokens_map[cidx+1] = widx
			widx += 1 

	return tokens_map 

def split_str(str_):
	# print(anlgy)
	split = re.split(r'(\W)', str_)

	# print(split)
	str_ = ' '.join([w for w in split if w!=' ' and w!=''])
	return str_


def read_data(path):
	cnt = 0
	with open(path, 'r') as f:
		squad_data = f.read()

	samples = {'id':[], 'tokens':[], 'ner_tags': [], 'concepts': []}

	label_list = ["O","B-Concept","I-Concept"]
	not_found_idx = []
	tot_cnt = 0
	for idx,row in enumerate(squad_data.split('\n')):
		if row.strip()!= '':
			tot_cnt += 1
			row_dict = json.loads(row.strip())
			anlgy = ' '.join(row_dict['analogy'].split())

			token_map = form_mapping_dict(anlgy)
			cncpts = row_dict['key'].lower().split('####')

			flag = True
			done_pos = []


			anlgy = split_str(anlgy)

			for cidx,c in enumerate(cncpts):
				cncpts[cidx] = split_str(c.lower())

			token_map = form_mapping_dict(anlgy)

			sample = {'id':idx,'tokens':anlgy.split(),'ner_tags':[0 for _ in anlgy.split()],'concepts':cncpts}

			for cncpt in cncpts:

				cncpt = cncpt
				anlgy = anlgy.lower() 
				rx = re.compile(r'(?<![^\W_]){}(?![^\W_])'.format(re.escape(cncpt)), re.I)
				
				found = False

				for m in rx.finditer(anlgy):
					found = True
					# print(m.span(),cncpt, m.start(),m.start()+len(cncpt))
				if not found:
					not_found_idx.append(idx)
					flag = False 

					break  

				for m in rx.finditer(anlgy):
					prev_done = False
					for c in range(m.span()[0], m.span()[1]):
						if c in done_pos:
							prev_done = True
							break 

					if not prev_done:
						done_pos += list(range(m.span()[0], m.span()[1]))
						break 
				
				if not prev_done:
					sample['ner_tags'][token_map[m.span()[0]]] = 1
					
					for c in range(m.span()[0]+1, m.span()[1]):
						try:
							sample['ner_tags'][token_map[c]] = 2
						except:
							pass		

			if flag:
				# for idx,s in enumerate(sample['ner_tags']):
				# 	if s != 0:
				# 		print(sample['tokens'][idx])
				samples['id'].append(sample['id'])
				samples['tokens'].append(sample['tokens'])
				samples['ner_tags'].append(sample['ner_tags'])
				samples['concepts'].append(sample['concepts'])
			else:
				cnt += 1
	# print(cnt, tot_cnt)  

	with open(path[:-6]+'_not_fnd_cncpt.txt','w') as f:
		for idx in not_found_idx:
			f.write(str(idx)+'\n')  
	return samples

def tokenize_and_align_labels(examples):
	tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, return_offsets_mapping=True)

	labels = []
	token_maps = []
	for i, label in enumerate(examples[f"ner_tags"]):
		token_maps.append(form_mapping_dict(' '.join(examples["tokens"][i])))
		word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:  # Set the special tokens to -100.
			if word_idx is None:
				label_ids.append(-100)
			elif word_idx != previous_word_idx:  # Only label the first token of a given word.
				label_ids.append(label[word_idx])
			else:
				label_ids.append(-100)
			previous_word_idx = word_idx
		labels.append(label_ids)

	tokenized_inputs["labels"] = labels

	return AnalogyDataset(tokenized_inputs)

if __name__ == '__main__':
	
	id2label = {0:"O", 1:"B-Concept", 2:"I-Concept"}
	label2id = {"O":0, "B-Concept":1, "I-Concept":2}

	for i in range(3):
		transformers.logging.set_verbosity_info()
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		#tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

		model = AutoModelForTokenClassification.from_pretrained(
	"bert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id)

		# model = AutoModelForTokenClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3, id2label=id2label, label2id=label2id)
	   
		data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

		train_samples = read_data('../data/split{}/train.jsonl'.format(i))
		test_samples = read_data('../data/split{}/test.jsonl'.format(i))

		train_dataset = tokenize_and_align_labels(train_samples)
		test_dataset = tokenize_and_align_labels(test_samples)

		seqeval = evaluate.load("seqeval")

		model_name = 'bert'
		lr = 5e-5

		training_args = TrainingArguments(
			output_dir="model_cncpt_{}_{}_{}".format(model_name,lr,i),
			learning_rate=lr,
			per_device_train_batch_size=8,
			per_device_eval_batch_size=8,
			num_train_epochs=3,
			weight_decay=0.01,
			evaluation_strategy="epoch",
			save_strategy="epoch"		   
		)

		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=test_dataset,
			tokenizer=tokenizer,
			data_collator=data_collator,
			compute_metrics=compute_metrics_tok
		)

		trainer.train()
		
		classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="average")
		ofile = 'opcncpt_bert{}'.format(i)
		print(compute_metrics_squad(classifier, test_samples, ofile))
	   
