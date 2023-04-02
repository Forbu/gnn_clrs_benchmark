"""
Script to test dm-clrs package
"""

import clrs


train_ds, num_samples, spec = clrs.create_dataset(
    folder="/tmp/CLRS30", algorithm="bfs", split="train", batch_size=32
)

for i, feedback in enumerate(train_ds.as_numpy_iterator()):
    print(feedback)
    print(dir(feedback))
    break
