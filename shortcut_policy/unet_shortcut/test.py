# test_shortcut_model.py
from shortcut_policy.shortcut_model import ShortcutModel

# Test instantiation
model = ShortcutModel(model=None, num_steps=1000, device="cuda")
print("ShortcutModel instance created successfully!")
