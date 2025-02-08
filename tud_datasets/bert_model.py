import torch
import torch.nn as nn
import pandas as pd
from numpy.random import RandomState
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, Trainer, TrainingArguments, BertModel 

sub_dataset1 = pd.read_csv("test.csv")
sub_dataset2 = pd.read_csv("spam.csv")

sub_dataset2 = sub_dataset2[["text", "score"]]

    
df = pd.concat([sub_dataset1, sub_dataset2], ignore_index=True)

rng = RandomState()

train = df.sample(frac=0.8, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

ds_train = Dataset.from_pandas(train)
ds_test = Dataset.from_pandas(test)

config = BertConfig.from_pretrained("bert-base-uncased", num_labels=1)
config.problem_type = "regression"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_fn(samples):
    return tokenizer(samples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset_train = ds_train.map(tokenize_fn, batched=True)

tokenized_dataset_train = tokenized_dataset_train.rename_column("score", "labels")
tokenized_dataset_train = tokenized_dataset_train.remove_columns([col for col in tokenized_dataset_train.column_names if col not in ["text", "labels", "input_ids", "attention_mask"]])
tokenized_dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

tokenized_dataset_test = ds_test.map(tokenize_fn, batched=True)

tokenized_dataset_test = tokenized_dataset_test.rename_column("score", "labels")
tokenized_dataset_test = tokenized_dataset_test.remove_columns([col for col in tokenized_dataset_test.column_names if col not in ["text", "labels", "input_ids", "attention_mask"]])
tokenized_dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



class BertForQualityRegression(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Regression head: map the [CLS] pooled output to one value.
        self.regressor = nn.Linear(config.hidden_size, 1)
        # Sigmoid activation to squash output between 0 and 1.
        self.sigmoid = nn.Sigmoid()
        # Initialize weights (using the parent's helper)
        self.init_weights()
 
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        # Apply sigmoid to get an output in (0, 1)
        probs = self.sigmoid(logits)
        loss = None
        if labels is not None:
            # Use MSELoss for regression
            loss_fct = nn.MSELoss()
            loss = loss_fct(probs.view(-1), labels.float())
        return {"loss": loss, "logits": probs}


model = BertForQualityRegression.from_pretrained("bert-base-uncased", config=config)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test
)


if __name__ == '__main__':
    trainer.train()
    model.save_pretrained("./quality_regression_model")
    tokenizer.save_pretrained("./quality_regression_model")

