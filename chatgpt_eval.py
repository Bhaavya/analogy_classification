import json 
import re 
from squad_eval import * 
import evaluate

def split_str(str_):
	# print(anlgy)
	split = re.split(r'(\W)', str_)

	# print(split)
	str_ = ' '.join([w for w in split if w!=' ' and w!=''])
	return str_

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

def eval_cncpt(all_pred_concepts, all_true_concepts):
	f1 = 0
	em = 0
	total = 0

	for idx,true_concepts in all_true_concepts.items():
		theoretical_answers = [{}]
		pred_answers = {}
		prev_index = -1 
		pred_concepts = []
		concept = ''
		total += 1
		try:
			pred_concepts = all_pred_concepts[idx]
		except:
			pred_concepts = []
		all_f1s = []
		all_ems = []
		# print('='*20)
		# print(pred_concepts)
		# print(true_concepts)
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

	f1 = 100.0 * f1/total
	em = 100.0 * em/total

	return {"f1":f1,"exact_match":em}

def eval_answer(all_pred_analogies, all_true_analogies):
	metric = evaluate.load("squad")
	theoretical_answers = [{"id": str(idx), "answers": {"text":[ans], "answer_start": [0]}} for idx,ans in all_true_analogies.items()]
	predicted_answers = []
	
	for k,v in all_true_analogies.items():
		try:
			predicted_answers.append({"id": str(k), "prediction_text": all_pred_analogies[k]})
		except:
			predicted_answers.append({"id": str(k), "prediction_text": ""})
	# print(theoretical_answers[:10], predicted_answers[:10])
	return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def eval_best_answer(all_pred_analogies, all_true_analogies):
	metric = evaluate.load("squad")
	
	f1 = 0.0
	em = 0.0
	total = 0
	for k,v in all_true_analogies.items():
		best_res = {'f1':-1,'em':-1}
		total += 1
		theoretical_answers = [{"id": str(k), "answers": {"text":[v], "answer_start": [0]}}]
		try:
			all_pred_analogies[k]
		except:
			continue 
		for preds in all_pred_analogies[k]:
			predicted_answers = []
			try:
				predicted_answers.append({"id": str(k), "prediction_text": preds})
			except:
				predicted_answers.append({"id": str(k), "prediction_text": ""})
	# print(theoretical_answers[:10], predicted_answers[:10])
			res = metric.compute(predictions=predicted_answers, references=theoretical_answers)
			# print(res, best_res['f1'])
			if res['f1']>best_res['f1']:
				best_res = {'f1':res['f1'],'em':res['exact_match']}
		f1 += best_res['f1']
		em += best_res['em']

	print('F1:', f1/total)
	print('EM:', em/total)
	return {'f1': f1/total, 'em':em/total}



def main():
	with open(not_fnd_idx_file) as f:
		data = f.read()

	not_fnd_idx = []

	for row in data.split('\n'):
		if row.strip() != '':
			not_fnd_idx.append(int(row.strip()))

	with open(f'{dir}/test.jsonl', 'r') as file:
		squad_data = file.read()

	all_true_concepts = {}
	all_true_analogies ={}
	for idx,row in enumerate(squad_data.split('\n')):
		if row.strip()!= '':
			row = json.loads(row.strip())
			if idx not in not_fnd_idx:
				cncpts = row['key'].lower().split('####')
				for cidx,c in enumerate(cncpts):
					cncpts[cidx] = split_str(c.lower())
				all_true_concepts[idx] = cncpts
			all_true_analogies[idx] = row['analogy']


	with open(outfile,'r') as f:
		data = f.read()

	all_pred_concepts = {}
	all_pred_analogies = {}
	for idx,row in enumerate(data.split('\n')):
		if row.strip()!= '':
			print(row)
			row = json.loads(row.strip('\n'))
			if int(row['idx'])-1 not in not_fnd_idx:
				try:
					src = row['source']
				except:
					src = ''
				try:
					target = row['target']
				except:
					target = ''
				all_pred_concepts[int(row['idx'])-1] = [src, target]
			
			try:
				anlgy = row['analogy']
			except:
				anlgy = ''
			try:
				all_pred_analogies[int(row['idx'])-1].append(anlgy)
			except:
				all_pred_analogies[int(row['idx'])-1] = [anlgy]

	print('Concept :', eval_cncpt(all_pred_concepts, all_true_concepts))
	print('Analogy :', eval_best_answer(all_pred_analogies, all_true_analogies))
			
if __name__ == '__main__':

	for split_no in range(1):
		dir = f'../data/splits/split{split_no+2}'
		outfile = f'{dir}/test_output2.jsonl'

		infile = ''
		not_fnd_idx_file = f'{dir}/test_not_fnd_cncpt.txt'
		main()