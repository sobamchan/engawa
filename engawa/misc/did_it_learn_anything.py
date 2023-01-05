from argparse import ArgumentParser

import numpy as np
import sienna
from numpy import dot
from numpy.linalg import norm

from engawa.model import EngawaModel


def cossim(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (norm(a) * norm(b))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt-path", type=str, required=True, help="Path to the pl checkpoint."
    )
    parser.add_argument(
        "--query-text",
        type=str,
        required=True,
        help="Query text to sort other texts by similality.",
    )
    parser.add_argument(
        "--pool-text-file",
        type=str,
        required=True,
        help="Line sep text file for testing.",
    )
    args = parser.parse_args()

    target_texts = sienna.load(args.pool_text_file)
    model = EngawaModel.load_from_checkpoint(args.ckpt_path)
    bart = model.bart.model
    tok = model.tokenizer

    query_vec = bart(**tok(args.query_text, return_tensors="pt")).last_hidden_state[0, 0, :].numpy()
    target_vecs = bart(**tok(target_texts, return_tensors="pt")).last_hidden_state[:, 0, :].numpy()

    cossims = []
    for i in range(len(target_texts)):
        cossims.append(cossim(query_vec, target_vecs[i, :]))

    print(cossims)
