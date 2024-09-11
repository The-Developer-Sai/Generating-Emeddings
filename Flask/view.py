import numpy as np

# Load the embeddings file
embeddings_filepath = 'embeddings/file-sample_500kB.docx_embeddings.npy'
embeddings = np.load(embeddings_filepath)

# Print the embeddings
print(embeddings)
