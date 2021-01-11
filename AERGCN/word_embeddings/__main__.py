from AERGCN.word_embeddings.embedders import Embedder, BatchEmbedder


def main():
    test_str = 'This is a test string for the word embedder.'
    test_str_2 = 'Another test string for the word embedder.'
    test_str_3 = 'The third test string.'

    test_tag = 'This'
    test_tag_2 = 'for'
    test_tag_3 = 'string'

    embedder = Embedder(model_path='xlm-roberta-base')
    embedder(test_str, test_tag, DEBUG=True)

    batch_embedder = BatchEmbedder(model_path='xlm-roberta-base')
    batch_embedder([test_str, test_str_2, test_str_3], [test_tag, test_tag_2, test_tag_3], DEBUG=True)


if __name__ == '__main__':
    main()
