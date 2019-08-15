# IA-NLP
## Inteligência artificial - Natural Language Processing (NLP)

Este projeto tem como objetivo criar um sistema para análise de sentimentos
utilizando a linguagem de programação **Python** e as bibliotecas
**NLTK**, **SciKit-Learn**, **Pandas** e **NumPy**.
O classificador vai receber textos e classifica-los entre as polaridades
**positivo**, **negativo** e **neutro**.

O classificador deve obter 65% de precisão.

### Instalação
1. Clone o repositório.
2. Crie uma venv com Python 3.7
3. Ative o virtualenv.
3. Instale as dependências.

```console
git clone https://github.com/LeonardoGazziro/IA-NLP.git
cd IA-NLP
python -m venv .IA-NLP
.\.IA-NLP\scripts\activate
pip install -r requirements.txt
```

Dentro do repositório será encontrado o arquivo _classificador.py_
com o código comentado.

Caso utilize o Jupyter NoteBook pode utilizar o arquivo _classificador.ipynb_.

### Dados do classificador.
O classificador utiliza um arquivo para treinar o algoritmo, _data_base.csv_, e um arquivo
 para realizar a classificação, _classificar.txt_.

Os dois arquivos foram criados utilizando o dataset **OpiSums-PT** disponibilizado pela 
**USP** no link: http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources

### Objetivo
O projeto foi criado para estudar o funcionamento de um classificador de emoções feito em Python, 
estudar IA e NLP.

