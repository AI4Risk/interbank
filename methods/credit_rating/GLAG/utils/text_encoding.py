
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import json
import torch
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F
from methods.credit_rating.GLAG.utils.tools import read_text, init_time_list

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_word2vec_embeddings(texts, model_path):

    print(f"Loading Word2Vec model from {model_path}...")
    try:

        if model_path.endswith('.bin') or model_path.endswith('.bin.gz'):
            wv = KeyedVectors.load_word2vec_format(model_path, binary=True)
        elif os.path.exists(model_path):

            try:
                wv = KeyedVectors.load(model_path)
            except:

                wv = KeyedVectors.load_word2vec_format(model_path, binary=False)
        else:
            print(f"Path not found locally, trying to load from gensim-data: {model_path}")
            wv = api.load(model_path)
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return []

    embeddings = []
    vector_size = wv.vector_size
    
    for text in tqdm(texts, desc="Encoding (Word2Vec)"):

        words = text.lower().split()
        word_vectors = []
        for word in words:
            if word in wv:
                word_vectors.append(wv[word])
        
        if len(word_vectors) > 0:

            sentence_emb = np.mean(word_vectors, axis=0)
        else:

            sentence_emb = np.zeros(vector_size)
            
        embeddings.append(torch.tensor(sentence_emb, dtype=torch.float32))
        
    return embeddings



def get_text_embeddings(input_texts: dict, method: str = 'Qwen', model_path: str = None, max_length: int = 512, batch_size: int = 32):

    
    keys = list(input_texts.keys())
    texts = list(input_texts.values())
    embeddings_dict = {}
    

    if method == 'word2vec':

        default_w2v_name = "GoogleNews-vectors-negative300.bin" 
        
        if model_path:
    
            if os.path.isdir(model_path):
                target_path = os.path.join(model_path, default_w2v_name)

            elif os.path.isdir(os.path.join(project_root, model_path)):
                target_path = os.path.join(project_root, model_path, default_w2v_name)
            else:
  
                target_path = model_path
        else:
      
            target_path = os.path.join(project_root, "word2vec", default_w2v_name)
            
        embeddings = get_word2vec_embeddings(texts, target_path)
        

        for j, key in enumerate(keys):
            if j < len(embeddings):
                emb = embeddings[j].unsqueeze(0) # (1, dim)
                emb = F.normalize(emb, p=2, dim=1)
                embeddings_dict[key] = emb.squeeze(0)
        return embeddings_dict

    # --- (Qwen, Finbert, Sentence-BERT) ---
    

    if method == 'Qwen':
        default_path = "embedding_models/" 
        padding_side = "left" 
        pool_type = "last_token"
        if max_length is None: max_length = 32768 
    elif method == 'Finbert':
        default_path = "FinBert/" 
        padding_side = "right"
        pool_type = "mean"
        if max_length is None: max_length = 512
    elif method == 'Sentence-BERT':
        default_path = "sentence-transformers/all-MiniLM-L6-v2" 
        padding_side = "right"
        pool_type = "mean"
        if max_length is None: max_length = 512
    else:
        raise ValueError(f"Unknown method: {method}")

    target_path = model_path if model_path else default_path
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {target_path} ({method}) on {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(target_path, padding_side=padding_side, trust_remote_code=True)
        model = AutoModel.from_pretrained(target_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model from {target_path}: {e}")
        return {}

    model.to(device)
    model.eval()


    total_samples = len(texts)
    print(f"Processing Text. Total samples: {total_samples}, Batch size: {batch_size}")


    for i in tqdm(range(0, total_samples, batch_size), desc=f"Encoding ({method})"):
        batch_keys = keys[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True, 
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad(): 
            outputs = model(**inputs)
            if pool_type == "last_token":
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            elif pool_type == "mean":
                embeddings = mean_pooling(outputs, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        for j, key in enumerate(batch_keys):
            embeddings_dict[key] = embeddings[j].cpu()

        del inputs, outputs, embeddings
        torch.cuda.empty_cache()

    return embeddings_dict
