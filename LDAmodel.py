#! /usr/bin/env python3

from __future__ import division

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import anndata
import scipy
import warnings
import sys, os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import argparse


def proc_raw(adata):
    # This is necessary to recover the raw, unnormalized data necessary for topic analysis
    #data = adata.raw[:, adata.to_df().columns.tolist()].to_adata().to_df()
    data_matrix = adata.raw[:, adata.var.index].to_adata().to_df().values
    return data_matrix


def LDA_model(data_matrix, n_components, doc_topic_prior, topic_word_prior, max_iter, learning_decay, learning_offset):
    lda = LatentDirichletAllocation(
        n_components=n_components, 
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        max_iter=max_iter, 
        random_state=43, 
        learning_method='online', 
        learning_decay=learning_decay,
        learning_offset=learning_offset
    )
    lda.fit(data_matrix)
    topic_matrix = lda.transform(data_matrix)
    w=lda.components_
    p = lda.perplexity(data_matrix)
    f = lda.score(data_matrix)
    return topic_matrix, w, p, f


def proc_adata_LDA(anndata, n_components, doc_topic_prior, topic_word_prior, max_iter, learning_decay, learning_offset, savedir, name):
    adata = sc.read_h5ad(anndata)
    data_matrix = proc_raw(adata)
    topic_matrix, w, p, f = LDA_model(data_matrix, n_components, doc_topic_prior, topic_word_prior, max_iter, learning_decay, learning_offset)
    pd.DataFrame([[p, f]], columns=['Perplexity', 'Log likelihood']).to_csv(os.path.join(savedir, name + '_stats.csv'), index=False)
    np.save(os.path.join(savedir, name + '_topic_matrix.npy'), topic_matrix)
    np.save(os.path.join(savedir, name + '_w_matrix.npy'), w)

    

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='script to run LDA')
    parser.add_argument('-a', '--anndata', required=True, type=str,
                        help = 'anndata matrix. Data can be processed, but raw data has to be stored')
    parser.add_argument('-n', '--ncomponents', required=True, type=int,
                        help = 'number of topics')
    parser.add_argument('-t', '--doc_topic_prior', default=0.1, type=float,
                        help = 'Sparsity of topics. Default 0.1')
    parser.add_argument('-w', '--topic_word_prior', default=0.001, type=float,
                        help = 'Sparsity of words. Default 0.001')
    parser.add_argument('-m', '--max_iter', default = 100, type=int,
                        help = 'Max iterations. Default 100')
    parser.add_argument('-d', '--learning_decay', default=0.5, type=float,
                        help = 'Learning_decay. Default 0.5')
    parser.add_argument('-o', '--learning_offset', default=50, type=float,
                        help = 'Learning_offset. Default 50.')
    parser.add_argument('-s', '--outdir', required=True, type=str,
                        help = 'Directory to write results')    
    parser.add_argument('-f', '--name', required=True, type=str,
                        help = 'Name of file to name numpy arrrays')
    args = parser.parse_args()
    status = proc_adata_LDA(
        args.anndata, args.ncomponents, args.doc_topic_prior, args.topic_word_prior, args.max_iter,
        args.learning_decay, args.learning_offset, args.outdir, args.name)