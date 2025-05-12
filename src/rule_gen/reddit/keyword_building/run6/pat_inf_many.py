


def reject_fn(ngram_score: dict[tuple, float],
              n1gram_score: dict[tuple, float],
              ):

    delta = 0.05
    for tokens, score in n1gram_score.items():
        pre = tuple(list(tokens)[:-1])
        post = tuple(list(tokens)[1:])

        skip = True
        if ngram_score[pre] + delta < score:
            skip = False

        if ngram_score[post] + delta < score:
            skip = False

        if not skip:
            yield tokens





