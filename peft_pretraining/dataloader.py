import itertools
import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        '''
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
        '''

        iter_data = iter(self.data)

        batch = []
        for example in iter_data:
            try:
                tokenized_example = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            except Exception as e:
                print(f"[Warning] Skipping example due to tokenizer error: {e}")
                continue  # skip

            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                try:
                    yield self._format_batch(batch)
                except Exception as e:
                    print(f"[Warning] Skipping batch due to formatting error: {e}")
                batch = []

        if batch:
            try:
                yield self._format_batch(batch)
            except Exception as e:
                print(f"[Warning] Skipping final batch due to formatting error: {e}")

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
