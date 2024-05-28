from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
import networkx as nx

# generate attention weights for input text using BERT model
def get_attention_map(text, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # attention tensors for each layer
    return attentions, tokenizer

# display attention weights as heatmap
def plot_attention_map(attentions, tokenizer, text):
    # select first layer and first head attention map
    attention = attentions[0][0, 0].detach().numpy()

    # tokenize input text
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # configure graph
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.xaxis.set_major_locator(FixedLocator(np.arange(len(tokens))))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(len(tokens))))
    ax.set_xticklabels(tokens, rotation=90, fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)

    # display the graph
    plt.show()

# display attention weights as connected graph
def plot_graph_from_attention(attentions, tokenizer, text):
    # select first layer and first head attention map
    attention = attentions[0][0, 0].detach().numpy()
    
    # tokenize input text
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # create connected graph from the attention matrix
    G = nx.DiGraph()

    # add nodes
    for i, token in enumerate(tokens):
        G.add_node(i, label=token)

    # add edges with the attention weights
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if attention[i, j] > 0.01: # ensure that edges are significant
                G.add_edge(i, j, weight=attention[i, j])

    # display the graph
    pos = nx.spring_layout(G)  # positions for nodes
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'))
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

#input text to analyze relation between tokens
text = "bob is the philanthropy chair"
attentions, tokenizer = get_attention_map(text)

#display attention matrix as a heat map
#this shows the weights between tokens
plot_attention_map(attentions, tokenizer, text)

#plot directed graph showing weights (edges) between tokens (vertices)
plot_graph_from_attention(attentions, tokenizer, text)