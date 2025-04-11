import torch

def compare_models(model1_path, model2_path):
    # Load models
    model1 = torch.load(model1_path, map_location=torch.device('cpu'))
    model2 = torch.load(model2_path, map_location=torch.device('cpu'))

    # Check if model architectures are the same
    if str(model1) != str(model2):
        print("Model architectures differ!")
        print("Model 1 architecture:")
        print(model1)
        print("\nModel 2 architecture:")
        print(model2)
        return

    print("Model architectures are identical.")

    # Check if model parameters are the same
    mismatches = []

    for (name1, param1), (name2, param2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        if name1 != name2:
            print(f"Parameter names differ: {name1} vs {name2}")
            mismatches.append((name1, name2))
            continue

        if not torch.equal(param1, param2):
            difference = torch.abs(param1 - param2).max().item()
            print(f"Difference in parameter: {name1}")
            print(f"Max difference: {difference}")
            mismatches.append((name1, difference))

    if not mismatches:
        print("The models are identical in both architecture and parameters.")
    else:
        print(f"\nTotal discrepancies found: {len(mismatches)}")
        for name, diff in mismatches:
            print(f"Discrepancy in {name}, max difference: {diff}")

# Example usage:
model1_path = 'model1.pth'
model2_path = 'model2.pth'
compare_models(model1_path, model2_path)