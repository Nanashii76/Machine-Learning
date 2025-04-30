# Twitter Disaster Detection with BERT

## Competição
**Nome**: [NLP - Getting Started](https://www.kaggle.com/c/nlp-getting-started)  
**Objetivo**: Classificar tweets em "desastre real" (1) ou "não desastre" (0) usando NLP.

---

## Solução Proposta
Classificador de alta precisão usando **BERT (Bidirectional Encoder Representations from Transformers)** com fine-tuning para classificação binária.

### Principais Características
- **Pré-processamento textual otimizado**
- **Transfer Learning com BERT-base-uncased**
- **Treinamento acelerado por GPU**
- **Validação com F1-score**
- **Pipeline completo de inferência**

---

## Performace

| Métrica |	Validação|
| F1-score | 81.3% |

--- 

## Configuração

1. Instalação
    > pip install torch pandas transformers scikit-learn
2. Estrutura de arquivos:
    ``` bash
    ├── data/
    │   ├── train.csv    # Dataset de treino
    │   └── test.csv     # Dataset de teste
    └── project.py  # Código principal
    ```

---

## Execução

> python disaster_tweets_bert.py 

Gera o arquivo submission.csv e imprime no terminal o f1-score


