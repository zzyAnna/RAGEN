import yaml

# Example config
config = {
    "training": {
        "gamma": 0.9,
        "learning_rate": 0.001
    }
}

def modify_nested_dict(config, fqn_key, value):
    entry_to_change = config
    keys = fqn_key.split(".")
    
    print(f"\nInitial entry_to_change points to: {id(entry_to_change)}")
    print(f"Config id: {id(config)}")
    
    # Navigate through the nested dictionary
    for i, k in enumerate(keys[:-1]):
        entry_to_change = entry_to_change[k]
        print(f"\nAfter accessing '{k}', entry_to_change points to: {id(entry_to_change)}")
        print(f"Value at this level: {entry_to_change}")
    
    # Change the final value
    print(f"\nBefore change, entry_to_change points to: {id(entry_to_change)}")
    entry_to_change[keys[-1]] = value
    print(f"After change, entry_to_change points to: {id(entry_to_change)}")
    return config

# Test the function
print("Original config:")
print(yaml.dump(config, default_flow_style=False))

# Try modifying training.gamma
config = modify_nested_dict(config, "training.gamma", 0.95)
print("\nFinal config:")
print(yaml.dump(config, default_flow_style=False)) 