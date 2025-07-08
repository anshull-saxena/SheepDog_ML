import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='Pretrained-LM', type=str)
parser.add_argument('--iters', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        aug1_encoding = self.tokenizer.encode_plus(
            aug_text1,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        aug2_encoding = self.tokenizer.encode_plus(
            aug_text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug2_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.tensor(fg_label, dtype=torch.float),
            'fg_label_aug1': torch.tensor(aug_fg1, dtype=torch.float),
            'fg_label_aug2': torch.tensor(aug_fg2, dtype=torch.float),
        }

    def __len__(self):
        return len(self.texts)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class RobertaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)
        output = self.fc_out(pooled_outputs)
        binary_output = self.binary_transform(pooled_outputs)
        return output, binary_output

def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size):
    ds = NewsDatasetAug(
        texts=contents,
        aug_texts1=contents_aug1,
        aug_texts2=contents_aug2,
        labels=np.array(labels),
        fg_label=fg_label,
        aug_fg1=aug_fg1,
        aug_fg2=aug_fg2,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        texts=contents,
        labels=np.array(labels),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):
    # Create directories for checkpoints and logs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load original and adversarial test sets (A, B, C, D)
    x_train, x_test, x_test_a, x_test_b, x_test_c, x_test_d, y_train, y_test = load_articles(datasetname)
    
    # Create DataLoaders for each test set
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)
    test_loader_a = create_eval_loader(x_test_a, y_test, tokenizer, max_len, batch_size)
    test_loader_b = create_eval_loader(x_test_b, y_test, tokenizer, max_len, batch_size)
    test_loader_c = create_eval_loader(x_test_c, y_test, tokenizer, max_len, batch_size)
    test_loader_d = create_eval_loader(x_test_d, y_test, tokenizer, max_len, batch_size)

    model = RobertaClassifier(n_classes=4).to(device)
    train_losses = []
    train_accs = []
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = 10000
    num_warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    for epoch in range(n_epochs):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(datasetname)
        train_loader = create_train_loader(
            x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size
        )

        avg_loss = []
        avg_acc = []

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            input_ids_aug1 = batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = batch_data["attention_mask_aug2"].to(device)
            targets = batch_data["labels"].to(device)
            fg_labels = batch_data["fg_label"].to(device)
            fg_labels_aug1 = batch_data["fg_label_aug1"].to(device)
            fg_labels_aug2 = batch_data["fg_label_aug2"].to(device)
            
            out_labels, out_labels_bi = model(input_ids=input_ids, attention_mask=attention_mask)
            out_labels_aug1, out_labels_bi_aug1 = model(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1)
            out_labels_aug2, out_labels_bi_aug2 = model(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2)
            fg_criterion = nn.BCELoss()
            finegrain_loss = (
                fg_criterion(F.sigmoid(out_labels), fg_labels) +
                fg_criterion(F.sigmoid(out_labels_aug1), fg_labels_aug1) +
                fg_criterion(F.sigmoid(out_labels_aug2), fg_labels_aug2)
            ) / 3

            out_probs = F.softmax(out_labels_bi, dim=-1)
            aug_log_prob1 = F.log_softmax(out_labels_bi_aug1, dim=-1)
            aug_log_prob2 = F.log_softmax(out_labels_bi_aug2, dim=-1)

            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(out_labels_bi, targets)

            cons_criterion = nn.KLDivLoss(reduction='batchmean')
            cons_loss = 0.5 * cons_criterion(aug_log_prob1, out_probs) + 0.5 * cons_criterion(aug_log_prob2, out_probs)
       
            loss = sup_loss + cons_loss + finegrain_loss

            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            _, pred = out_labels_bi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        print(f"Iter {iter:03d} | Epoch {epoch+1:05d} | Train Loss {np.mean(avg_loss):.4f} | Train Acc. {np.mean(avg_acc):.4f}")

        if epoch == n_epochs - 1:
            model.eval()
            y_pred = []
            y_pred_a = []
            y_pred_b = []
            y_pred_c = []
            y_pred_d = []
            y_test = []

            for batch_data in tqdm(test_loader, desc="Evaluating Original"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    targets = batch_data["labels"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)
                    y_pred.append(val_pred)
                    y_test.append(targets)

            for batch_data in tqdm(test_loader_a, desc="Evaluating A (Objective)"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)
                    y_pred_a.append(val_pred)

            for batch_data in tqdm(test_loader_b, desc="Evaluating B (Neutral)"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)
                    y_pred_b.append(val_pred)

            for batch_data in tqdm(test_loader_c, desc="Evaluating C (Emotionally Triggering)"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)
                    y_pred_c.append(val_pred)

            for batch_data in tqdm(test_loader_d, desc="Evaluating D (Sensational)"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, val_pred = val_out.max(dim=1)
                    y_pred_d.append(val_pred)

            y_pred = torch.cat(y_pred, dim=0)
            y_pred_a = torch.cat(y_pred_a, dim=0)
            y_pred_b = torch.cat(y_pred_b, dim=0)
            y_pred_c = torch.cat(y_pred_c, dim=0)
            y_pred_d = torch.cat(y_pred_d, dim=0)
            y_test = torch.cat(y_test, dim=0)

            acc = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
            precision, recall, fscore, _ = score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            acc_a = accuracy_score(y_test.cpu().numpy(), y_pred_a.cpu().numpy())
            precision_a, recall_a, fscore_a, _ = score(y_test.cpu().numpy(), y_pred_a.cpu().numpy(), average='macro')
            acc_b = accuracy_score(y_test.cpu().numpy(), y_pred_b.cpu().numpy())
            precision_b, recall_b, fscore_b, _ = score(y_test.cpu().numpy(), y_pred_b.cpu().numpy(), average='macro')
            acc_c = accuracy_score(y_test.cpu().numpy(), y_pred_c.cpu().numpy())
            precision_c, recall_c, fscore_c, _ = score(y_test.cpu().numpy(), y_pred_c.cpu().numpy(), average='macro')
            acc_d = accuracy_score(y_test.cpu().numpy(), y_pred_d.cpu().numpy())
            precision_d, recall_d, fscore_d, _ = score(y_test.cpu().numpy(), y_pred_d.cpu().numpy(), average='macro')

            acc_res = (acc_a + acc_b + acc_c + acc_d) / 4
            precision_res = (precision_a + precision_b + precision_c + precision_d) / 4
            recall_res = (recall_a + recall_b + recall_c + recall_d) / 4
            fscore_res = (fscore_a + fscore_b + fscore_c + fscore_d) / 4

    torch.save(model.state_dict(), os.path.join('checkpoints', f'{datasetname}_iter{iter}.m'))

    print(f"-----------------End of Iter {iter:03d}-----------------")
    print("Original:")
    print([f'Global Test Accuracy: {acc:.4f}',
           f'Precision: {precision:.4f}',
           f'Recall: {recall:.4f}',
           f'F1: {fscore:.4f}'])
    print("Adversarial (A - Objective):")
    print([f'Global Test Accuracy: {acc_a:.4f}',
           f'Precision: {precision_a:.4f}',
           f'Recall: {recall_a:.4f}',
           f'F1: {fscore_a:.4f}'])
    print("Adversarial (B - Neutral):")
    print([f'Global Test Accuracy: {acc_b:.4f}',
           f'Precision: {precision_b:.4f}',
           f'Recall: {recall_b:.4f}',
           f'F1: {fscore_b:.4f}'])
    print("Adversarial (C - Emotionally Triggering):")
    print([f'Global Test Accuracy: {acc_c:.4f}',
           f'Precision: {precision_c:.4f}',
           f'Recall: {recall_c:.4f}',
           f'F1: {fscore_c:.4f}'])
    print("Adversarial (D - Sensational):")
    print([f'Global Test Accuracy: {acc_d:.4f}',
           f'Precision: {precision_d:.4f}',
           f'Recall: {recall_d:.4f}',
           f'F1: {fscore_d:.4f}'])
    print("Adversarial (Average):")
    print([f'Global Test Accuracy: {acc_res:.4f}',
           f'Precision: {precision_res:.4f}',
           f'Recall: {recall_res:.4f}',
           f'F1: {fscore_res:.4f}'])

    return (acc, precision, recall, fscore,
            acc_res, precision_res, recall_res, fscore_res,
            acc_a, precision_a, recall_a, fscore_a,
            acc_b, precision_b, recall_b, fscore_b,
            acc_c, precision_c, recall_c, fscore_c,
            acc_d, precision_d, recall_d, fscore_d)

datasetname = args.dataset_name
batch_size = args.batch_size
max_len = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
n_epochs = args.n_epochs
iterations = args.iters

os.makedirs('logs', exist_ok=True)

test_accs = []
prec_all, rec_all, f1_all = [], [], []
test_accs_a, prec_all_a, rec_all_a, f1_all_a = [], [], [], []
test_accs_b, prec_all_b, rec_all_b, f1_all_b = [], [], [], []
test_accs_c, prec_all_c, rec_all_c, f1_all_c = [], [], [], []
test_accs_d, prec_all_d, rec_all_d, f1_all_d = [], [], [], []
test_accs_res = []
prec_all_res, rec_all_res, f1_all_res = [], [], []

for iter in range(iterations):
    set_seed(iter)
    metrics = train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter)
    
    (acc, precision, recall, f1,
     acc_res, precision_res, recall_res, f1_res,
     acc_a, precision_a, recall_a, f1_a,
     acc_b, precision_b, recall_b, f1_b,
     acc_c, precision_c, recall_c, f1_c,
     acc_d, precision_d, recall_d, f1_d) = metrics

    test_accs.append(acc)
    prec_all.append(precision)
    rec_all.append(recall)
    f1_all.append(f1)
    test_accs_res.append(acc_res)
    prec_all_res.append(precision_res)
    rec_all_res.append(recall_res)
    f1_all_res.append(f1_res)
    test_accs_a.append(acc_a)
    prec_all_a.append(precision_a)
    rec_all_a.append(recall_a)
    f1_all_a.append(f1_a)
    test_accs_b.append(acc_b)
    prec_all_b.append(precision_b)
    rec_all_b.append(recall_b)
    f1_all_b.append(f1_b)
    test_accs_c.append(acc_c)
    prec_all_c.append(precision_c)
    rec_all_c.append(recall_c)
    f1_all_c.append(f1_c)
    test_accs_d.append(acc_d)
    prec_all_d.append(precision_d)
    rec_all_d.append(recall_d)
    f1_all_d.append(f1_d)

print(f"Total_Test_Accuracy: {sum(test_accs)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all)/iterations:.4f}")
print(f"Restyle_A_Test_Accuracy: {sum(test_accs_a)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all_a)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all_a)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all_a)/iterations:.4f}")
print(f"Restyle_B_Test_Accuracy: {sum(test_accs_b)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all_b)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all_b)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all_b)/iterations:.4f}")
print(f"Restyle_C_Test_Accuracy: {sum(test_accs_c)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all_c)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all_c)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all_c)/iterations:.4f}")
print(f"Restyle_D_Test_Accuracy: {sum(test_accs_d)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all_d)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all_d)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all_d)/iterations:.4f}")
print(f"Restyle_Average_Test_Accuracy: {sum(test_accs_res)/iterations:.4f}|"
      f"Prec_Macro: {sum(prec_all_res)/iterations:.4f}|"
      f"Rec_Macro: {sum(rec_all_res)/iterations:.4f}|"
      f"F1_Macro: {sum(f1_all_res)/iterations:.4f}")

log_file = os.path.join('logs', f'log_{datasetname}_{args.model_name}_iter{iterations}.txt')
with open(log_file, 'a+') as f:
    f.write('-------------Original-------------\n')
    f.write(f'All Acc.s:{test_accs}\n')
    f.write(f'All Prec.s:{prec_all}\n')
    f.write(f'All Rec.s:{rec_all}\n')
    f.write(f'All F1.s:{f1_all}\n')
    f.write(f'Average acc.: {sum(test_accs) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all) / iterations}, {sum(rec_all) / iterations}, {sum(f1_all) / iterations}\n')

    f.write('\n-------------Adversarial (A - Objective)------------\n')
    f.write(f'All Acc.s:{test_accs_a}\n')
    f.write(f'All Prec.s:{prec_all_a}\n')
    f.write(f'All Rec.s:{rec_all_a}\n')
    f.write(f'All F1.s:{f1_all_a}\n')
    f.write(f'Average acc.: {sum(test_accs_a) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all_a) / iterations}, {sum(rec_all_a) / iterations}, {sum(f1_all_a) / iterations}\n')

    f.write('\n-------------Adversarial (B - Neutral)------------\n')
    f.write(f'All Acc.s:{test_accs_b}\n')
    f.write(f'All Prec.s:{prec_all_b}\n')
    f.write(f'All Rec.s:{rec_all_b}\n')
    f.write(f'All F1.s:{f1_all_b}\n')
    f.write(f'Average acc.: {sum(test_accs_b) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all_b) / iterations}, {sum(rec_all_b) / iterations}, {sum(f1_all_b) / iterations}\n')

    f.write('\n-------------Adversarial (C - Emotionally Triggering)------------\n')
    f.write(f'All Acc.s:{test_accs_c}\n')
    f.write(f'All Prec.s:{prec_all_c}\n')
    f.write(f'All Rec.s:{rec_all_c}\n')
    f.write(f'All F1.s:{f1_all_c}\n')
    f.write(f'Average acc.: {sum(test_accs_c) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all_c) / iterations}, {sum(rec_all_c) / iterations}, {sum(f1_all_c) / iterations}\n')

    f.write('\n-------------Adversarial (D - Sensational)------------\n')
    f.write(f'All Acc.s:{test_accs_d}\n')
    f.write(f'All Prec.s:{prec_all_d}\n')
    f.write(f'All Rec.s:{rec_all_d}\n')
    f.write(f'All F1.s:{f1_all_d}\n')
    f.write(f'Average acc.: {sum(test_accs_d) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all_d) / iterations}, {sum(rec_all_d) / iterations}, {sum(f1_all_d) / iterations}\n')

    f.write('\n-------------Adversarial (Average)------------\n')
    f.write(f'All Acc.s:{test_accs_res}\n')
    f.write(f'All Prec.s:{prec_all_res}\n')
    f.write(f'All Rec.s:{rec_all_res}\n')
    f.write(f'All F1.s:{f1_all_res}\n')
    f.write(f'Average acc.: {sum(test_accs_res) / iterations}\n')
    f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all_res) / iterations}, {sum(rec_all_res) / iterations}, {sum(f1_all_res) / iterations}\n')