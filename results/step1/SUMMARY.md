# HW4.a Step 1: BERT fine-tune summary

- GPU: NVIDIA GeForce RTX 4090
- Row range used (train split, inclusive): 0 to 4999 (n=5000)
- Seed: 2122
- Split: train=3500 val=750 test=750
- Settings: {"epochs": 4, "batch_size": 32, "lr": 2e-05, "max_len": 64, "precision": "fp32", "model": "bert-base-uncased"}

## Per-epoch

| Epoch | Train loss | Val loss | Val acc | Epoch seconds |
|---|---|---|---|---|
| 1 | 0.8894 | 0.7215 | 0.6707 | 5.75 |
| 2 | 0.6227 | 0.7060 | 0.6853 | 5.28 |
| 3 | 0.4059 | 0.8449 | 0.6733 | 5.25 |
| 4 | 0.2240 | 0.9549 | 0.6707 | 5.14 |

- Best val accuracy at epoch 2; lowest val loss at epoch 2
- Epochs trained past best: 2 (cost 10.4 s)
- Test loss 0.7535, test accuracy 0.6653 (best-val-accuracy checkpoint)

## Confusion matrix (rows true, cols predicted: negative, neutral, positive)

```
[92, 24, 4]
[69, 220, 47]
[22, 85, 187]
```

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.503 | 0.767 | 0.607 | 120 |
| neutral | 0.669 | 0.655 | 0.662 | 336 |
| positive | 0.786 | 0.636 | 0.703 | 294 |

Worst class (lowest F1): **negative** (28 misclassified in test)

## Five misclassified examples from the worst class

- true=negative, predicted=positive: Hope David Cameron enjoys his breakfast tomorrow and isn't put off thinking about all the people he starves
- true=negative, predicted=neutral: Found FOX NEWS Poll before debate without Rick Perry on it, this maybe why  Polls rigged? VIP
- true=negative, predicted=neutral: "Stoops admits he wouldn't believe they'd be 2nd in the country in interceptions. """"I gotta be honest there."""""
- true=negative, predicted=neutral: i may have forgotten all the parts in Ultron that didn't involve Chris Evans' ass. is there still a JARVIS in Tony's smart home/suit
- true=negative, predicted=positive: @user do u enjoy his 2nd rate Michael Jackson bit? Honest ques. Like the can't feel face song but god it's so obvious they want MJ 2.0

## HW4.5 baseline measurements

- Peak GPU memory allocated: 2.466 GB
- Peak GPU memory reserved: 3.027 GB
- Wall-clock training time: 23.03 s

## Custom inference after save and reload

- "I absolutely love this new phone, best purchase ever!" -> positive
- "Worst customer service I have ever had. Never again." -> negative
- "The meeting is scheduled for 3pm tomorrow." -> neutral
