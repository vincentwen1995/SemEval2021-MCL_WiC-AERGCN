from torch.utils.data import DataLoader

from AERGCN.data.datasets import MCL_WiC_Dataset
from AERGCN.word_embeddings.embedders import BatchEmbedder


def main():
    batch_embedder = BatchEmbedder(model_path='xlm-roberta-large')
    dataset_train = MCL_WiC_Dataset(
        split='training',
        lang_1='en',
        lang_2='en',
        include_pos_tags=False,
        batch_embedder=batch_embedder,
        k=20,
    )
    print(len(dataset_train))
    print(dataset_train[len(dataset_train) - 1])

    dataset_dev = MCL_WiC_Dataset(
        split='development',
        lang_1='en',
        lang_2='en',
        include_pos_tags=False,
        batch_embedder=batch_embedder,
        k=20,
    )
    print(len(dataset_dev))
    print(dataset_dev[len(dataset_dev) - 1])

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
        pin_memory=True,
    )

    for i, batch_outputs_dict in enumerate(dataloader_train):
        print(batch_outputs_dict.keys())
        if i > 1:
            break


if __name__ == '__main__':
    main()
