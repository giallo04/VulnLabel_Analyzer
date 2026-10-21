# Report di validazione manuale

- Righe annotate valide: **43** (non compilate: 0, errori: 2)
- Convenzione: ok = vulnerabile, no = non vulnerabile; classe positiva = VULNERABILE

## Matrice di confusione complessiva (non pesata)

| | Umano: vulnerabile | Umano: non vulnerabile |
|---|---|---|
| **LLM: vulnerabile** | TP = 9 | FP = 1 |
| **LLM: non vulnerabile** | FN = 1 | TN = 32 |

## Metriche per categoria e totali

| Gruppo | n | TP | FP | FN | TN | Acc | Prec | Rec | Spec | NPV | F1 | F1-neg | BalAcc | MCC | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CORRECT_LABEL | 10 | 9 | 1 | 0 | 0 | 0.900 | 0.900 | 1.000 | 0.000 | n/a | 0.947 | n/a | 0.500 | n/a | 0.000 |
| WRONG_LABEL_CONSISTENCY | 10 | 0 | 0 | 0 | 10 | 1.000 | n/a | n/a | 1.000 | 1.000 | n/a | 1.000 | n/a | n/a | n/a |
| WRONG_LABEL_FEATURE | 5 | 0 | 0 | 0 | 5 | 1.000 | n/a | n/a | 1.000 | 1.000 | n/a | 1.000 | n/a | n/a | n/a |
| WRONG_LABEL_IRRELEVANT | 9 | 0 | 0 | 1 | 8 | 0.889 | n/a | 0.000 | 1.000 | 0.889 | n/a | 0.941 | 0.500 | n/a | 0.000 |
| WRONG_LABEL_SPREAD | 9 | 0 | 0 | 0 | 9 | 1.000 | n/a | n/a | 1.000 | 1.000 | n/a | 1.000 | n/a | n/a | n/a |
| TOTALE (non pesato) | 43 | 9 | 1 | 1 | 32 | 0.953 | 0.900 | 0.900 | 0.970 | 0.970 | 0.900 | 0.970 | 0.935 | 0.870 | 0.870 |

`n/a` = metrica non definita (denominatore nullo): in una categoria che il tool mappa sempre sulla stessa classe, precision/recall/specificity/NPV sono in parte indefinite; per queste categorie l'indicatore significativo e' l'accuracy (= quota di verdetti LLM confermati dall'analisi manuale).

## Intervalli di confidenza 95% (Wilson)

| Gruppo | Accuracy | Precision | Recall | Specificity | NPV |
|---|---|---|---|---|---|
| CORRECT_LABEL | 0.900 [0.596, 0.982] | 0.900 [0.596, 0.982] | 1.000 [0.701, 1.000] | 0.000 [0.000, 0.793] | n/a n/a |
| WRONG_LABEL_CONSISTENCY | 1.000 [0.722, 1.000] | n/a n/a | n/a n/a | 1.000 [0.722, 1.000] | 1.000 [0.722, 1.000] |
| WRONG_LABEL_FEATURE | 1.000 [0.566, 1.000] | n/a n/a | n/a n/a | 1.000 [0.566, 1.000] | 1.000 [0.566, 1.000] |
| WRONG_LABEL_IRRELEVANT | 0.889 [0.565, 0.980] | n/a n/a | 0.000 [0.000, 0.793] | 1.000 [0.676, 1.000] | 0.889 [0.565, 0.980] |
| WRONG_LABEL_SPREAD | 1.000 [0.701, 1.000] | n/a n/a | n/a n/a | 1.000 [0.701, 1.000] | 1.000 [0.701, 1.000] |
| TOTALE (non pesato) | 0.953 [0.845, 0.987] | 0.900 [0.596, 0.982] | 0.900 [0.596, 0.982] | 0.970 [0.847, 0.995] | 0.970 [0.847, 0.995] |

## Metriche aggiuntive (totale non pesato)

- False positive rate: 0.030
- False negative rate: 0.100
- Prevalenza vulnerabili (umano): 0.233

## Rilevamento del rumore nel dataset

Rumore = funzione etichettata vulnerabile (1) ma non vulnerabile secondo l'analisi manuale. Per il rilevamento del rumore: precision = NPV, recall = specificity.

- Rumore stimato dall'analisi manuale (campione): 0.767 [0.623, 0.868]  (campione stratificato: non e' una stima della popolazione)
- Precision rilevamento rumore (NPV): 0.970 [0.847, 0.995]
- Recall rilevamento rumore (specificity): 0.970 [0.847, 0.995]

## Disaccordi LLM vs umano (2)

| id | project | funzione | categoria LLM | LLM | umano | link | note |
|---|---|---|---|---|---|---|---|
| 14 | linux | `ext2_xattr_cache_find` | CORRECT_LABEL | VULNERABILE | non vulnerabile | https://github.com/torvalds/linux/commit/be0726d33cb8f411945884664924bed3cb8c70ee |  |
| 30 | ImageMagick | `ScaleQuantumToLong` | WRONG_LABEL_IRRELEVANT | NON_VULNERABILE | vulnerabile | https://github.com/ImageMagick/ImageMagick/commit/95d4e94e0353e503b71a53f5e6fad173c7c70c90 |  |
