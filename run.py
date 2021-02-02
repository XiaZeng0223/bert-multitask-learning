import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)

from bert_multitask_learning import train_bert_multitask, eval_bert_multitask, predict_bert_multitask
from bert_multitask_learning.preproc_decorator import preprocessing_fn
from bert_multitask_learning.params import BaseParams
import bert_multitask_learning

from sklearn.metrics import f1_score, precision_score, recall_score

import argparse
import jsonlines
import random
import numpy as np
import pprint


from torch.utils.data import Dataset





parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='/data//home/acw560/two-step/data/corpus.jsonl')
parser.add_argument('--train', type=str, default='/data//home/acw560/two-step/data/claims_train.jsonl')
parser.add_argument('--dev', type=str, default='/data/home/acw560/two-step/data/claims_dev.jsonl')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny')
parser.add_argument('--model', type=str, default='TFAutoModel')
parser.add_argument('--config', type=str, default='BertConfig')
parser.add_argument('--tokenizer', type=str, default='BertTokenizer')
parser.add_argument('--init_lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()



class ScifactDataset(Dataset):
    def __init__(self, corpus: str, claims: str, objective: str):
        self.samples = []

        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        if objective=='neutral':
            # binary classification: merge SUPPORT AND CONTRADICT as 0, NOT_ENOUGH_INFO as 1.
            # neutral detection
            label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 0}
        elif objective=='entailment':
            # binary classification: merge NOT_ENOUGH_INFO AND CONTRADICT as 0, SUPPORT as 1.
            # entailment detection
            label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 0, 'SUPPORT': 1}
        elif objective=='contradict':
            # binary classification: merge NOT_ENOUGH_INFO AND SUPPORT as 0, CONTRADICT as 1.
            # contradict detection
            label_encodings = {'CONTRADICT': 1, 'NOT_ENOUGH_INFO': 0, 'SUPPORT': 0}
        else:
            # three-way classification
            label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}


        for claim in jsonlines.open(claims):
            if claim['evidence']:
                for doc_id, evidence_sets in claim['evidence'].items():
                    doc = corpus[int(doc_id)]

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]  # directly use the first evidence set label
                        # because currently all evidence sets have
                        # the same label
                    })

                    # Add negative samples
                    non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(1, 2), len(non_rationale_idx)))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
            else:
                # Add negative samples
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



@preprocessing_fn
def neutral_cls(params: BaseParams, mode: str):
    "Simple example to demonstrate singe modal tuple of list return"
    trainset = ScifactDataset(args.corpus, args.train, 'neutral')
    devset = ScifactDataset(args.corpus, args.dev, 'neutral')
    if mode == bert_multitask_learning.TRAIN:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in trainset]
        target = [x['label'] for x in trainset]
    else:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in devset]
        target = [x['label'] for x in devset]
    return input, target

@preprocessing_fn
def entailment_cls(params: BaseParams, mode: str):
    "Simple example to demonstrate singe modal tuple of list return"
    trainset = ScifactDataset(args.corpus, args.train, 'entailment')
    devset = ScifactDataset(args.corpus, args.dev, 'entailment')
    if mode == bert_multitask_learning.TRAIN:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in trainset]
        target = [x['label'] for x in trainset]
    else:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in devset]
        target = [x['label'] for x in devset]
    return input, target

@preprocessing_fn
def contradict_cls(params: BaseParams, mode: str):
    "Simple example to demonstrate singe modal tuple of list return"
    trainset = ScifactDataset(args.corpus, args.train, 'contradict')
    devset = ScifactDataset(args.corpus, args.dev, 'contradict')
    if mode == bert_multitask_learning.TRAIN:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in trainset]
        target = [x['label'] for x in trainset]
    else:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in devset]
        target = [x['label'] for x in devset]
    return input, target


@preprocessing_fn
def three_cls(params: BaseParams, mode: str):
    "Simple example to demonstrate singe modal tuple of list return"
    trainset = ScifactDataset(args.corpus, args.train, 'three')
    devset = ScifactDataset(args.corpus, args.dev, 'three')
    if mode == bert_multitask_learning.TRAIN:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in trainset]
        target = [x['label'] for x in trainset]
    else:
        input = [x['claim']+'[SEP]'+x['rationale'] for x in devset]
        target = [x['label'] for x in devset]
    return input, target


def eval(preds_, task):
    # contradict_cls
    devset = ScifactDataset(args.corpus, args.dev, task)
    targets = [x['label'] for x in devset]
    outputs=np.argmax(preds_, axis=1)
    print('Evaluating {} cls.'.format(task))
    # print(inputs)
    # print(type(targets), len(targets))
    # print(targets)
    # print(type(outputs), len(outputs))
    # print(outputs)

    return "'macro_f1': {}  'f1': {}  'precision': {}  'recall': {}".format(f1_score(targets, outputs, zero_division=0, average='macro'),
                                                                                 tuple(f1_score(targets, outputs, zero_division=0, average=None)),
                                                                                tuple(precision_score(targets, outputs, zero_division=0, average=None)),
                                                                                tuple(recall_score(targets, outputs, zero_division=0, average=None)))

if __name__ == '__main__':


    params = BaseParams()


    #model choice
    # specify model and its loading module
    params.transformer_model_name = args.model_name
    params.transformer_model_loading = args.model
    # specify tokenizer and its loading module
    params.transformer_tokenizer_name = args.model_name
    params.transformer_tokenizer_loading = args.tokenizer
    # specify config and its loading module
    params.transformer_config_name = args.model_name
    params.transformer_config_loading = args.config


    params.init_lr = args.init_lr
    params.batch_size = args.batch_size

    # params.ckpt_dir='/home/zeng/bert-multitask-learning/models/contradict_cls_entailment_cls_neutral_cls_three_cls_ckpt'
 
    problem = 'neutral_cls|entailment_cls|contradict_cls|three_cls'

    processing_fn_dict = {'neutral_cls': neutral_cls, 'entailment_cls': entailment_cls, 'contradict_cls': contradict_cls, 'three_cls': three_cls }
    problem_type_dict = {'neutral_cls': 'cls', 'entailment_cls': 'cls', 'contradict_cls': 'cls', 'three_cls': 'cls'}

    # train
    model = train_bert_multitask(
        problem=problem,
        num_epochs=args.epochs,
        problem_type_dict=problem_type_dict,
        processing_fn_dict=processing_fn_dict,
        continue_training=True,
        params=params   #pass params
    )

    # # eval
    # print('evaluating... \n \n')
    # eval_dict = eval_bert_multitask(
    #     problem=problem,
    #     problem_type_dict=problem_type_dict,
    #     processing_fn_dict=processing_fn_dict,
    #     model_dir=params.ckpt_dir)
    # print(eval_dict)



    # eval each task
    print('Evaluating... \n \n')
    devset = ScifactDataset(args.corpus, args.dev, 'three')
    inputs = [x['claim']+'[SEP]'+x['rationale'] for x in devset]
    preds = predict_bert_multitask(
        problem=problem,
        inputs=inputs,
        model_dir=params.ckpt_dir,
        problem_type_dict=problem_type_dict,
        processing_fn_dict=processing_fn_dict,
        return_model=False)

    results={}


    # three_cls
    results['three'] = eval(preds_=preds['three_cls'], task='three')

    # contradict_cls
    results['contradict'] = eval(preds_=preds['contradict_cls'], task='contradict')

    # entailment_cls
    results['entailment'] = eval(preds_=preds['entailment_cls'], task='entailment')

    # neutral_cls
    results['neutral'] = eval(preds_=preds['neutral_cls'], task='neutral')

    pprint.pprint(results)
