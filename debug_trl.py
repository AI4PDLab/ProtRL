from trl import GRPOTrainer
import inspect

print("Methods in GRPOTrainer:")
for name, method in inspect.getmembers(GRPOTrainer, predicate=inspect.isfunction):
    print(name)

from trl.trainer.utils import shuffle_sequence_dict
print("\nSource of shuffle_sequence_dict:")
try:
    print(inspect.getsource(shuffle_sequence_dict))
except Exception as e:
    print(f"Could not get source: {e}")
