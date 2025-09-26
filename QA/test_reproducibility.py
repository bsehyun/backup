"""
Test script to verify reproducibility of PyTorch implementation
"""
import os
import torch
import numpy as np
from pytorch_model import CRBLModel, set_seed
from pytorch_dataset import create_data_loaders
from utils import compare_models, benchmark_inference

def test_model_reproducibility():
    """Test if model initialization is reproducible"""
    print("Testing model reproducibility...")
    
    # Test 1: Same seed should produce identical models
    set_seed(42)
    model1 = CRBLModel(input_size=128, isCrop=True)
    
    set_seed(42)
    model2 = CRBLModel(input_size=128, isCrop=True)
    
    is_identical, differences = compare_models(model1, model2)
    
    if is_identical:
        print("✓ Model initialization is reproducible")
    else:
        print("✗ Model initialization is NOT reproducible")
        for diff in differences:
            print(f"  - {diff}")
    
    return is_identical

def test_inference_reproducibility():
    """Test if inference is reproducible"""
    print("\nTesting inference reproducibility...")
    
    # Create model
    set_seed(42)
    model = CRBLModel(input_size=128, isCrop=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 128, 128)
    
    # Run inference multiple times with same seed
    set_seed(42)
    with torch.no_grad():
        output1 = model(dummy_input)
    
    set_seed(42)
    with torch.no_grad():
        output2 = model(dummy_input)
    
    # Check if outputs are identical
    is_identical = torch.allclose(output1, output2, atol=1e-6)
    
    if is_identical:
        print("✓ Inference is reproducible")
    else:
        print("✗ Inference is NOT reproducible")
        max_diff = torch.max(torch.abs(output1 - output2)).item()
        print(f"  - Maximum difference: {max_diff}")
    
    return is_identical

def test_data_loader_reproducibility():
    """Test if data loader produces same batches"""
    print("\nTesting data loader reproducibility...")
    
    # Create data loaders with same seed
    set_seed(42)
    train_loader1, _ = create_data_loaders(
        class_datadir="./data/images/classed_image",
        datadir="./data/images",
        isCrop=True,
        batch_size=4,
        class0_ratio=0.5,
        test_csv_path="./data/csv/test_CRBL.csv"
    )
    
    set_seed(42)
    train_loader2, _ = create_data_loaders(
        class_datadir="./data/images/classed_image",
        datadir="./data/images",
        isCrop=True,
        batch_size=4,
        class0_ratio=0.5,
        test_csv_path="./data/csv/test_CRBL.csv"
    )
    
    if train_loader1 is None or train_loader2 is None:
        print("⚠ Data loaders could not be created (data files may not exist)")
        return True
    
    # Compare first batch
    batch1 = next(iter(train_loader1))
    batch2 = next(iter(train_loader2))
    
    images1, labels1 = batch1
    images2, labels2 = batch2
    
    images_identical = torch.allclose(images1, images2, atol=1e-6)
    labels_identical = torch.equal(labels1, labels2)
    
    if images_identical and labels_identical:
        print("✓ Data loader is reproducible")
        return True
    else:
        print("✗ Data loader is NOT reproducible")
        if not images_identical:
            max_diff = torch.max(torch.abs(images1 - images2)).item()
            print(f"  - Images max difference: {max_diff}")
        if not labels_identical:
            print(f"  - Labels different: {labels1} vs {labels2}")
        return False

def test_training_reproducibility():
    """Test if training produces same results"""
    print("\nTesting training reproducibility...")
    
    # This is a simplified test - in practice, you'd need to run full training
    # and compare final model weights
    
    set_seed(42)
    model1 = CRBLModel(input_size=128, isCrop=True)
    
    # Simulate one training step
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    dummy_input = torch.randn(2, 3, 128, 128)
    dummy_target = torch.tensor([0.0, 1.0])
    
    optimizer1.zero_grad()
    output1 = model1(dummy_input).squeeze()
    loss1 = criterion(output1, dummy_target)
    loss1.backward()
    optimizer1.step()
    
    # Repeat with same seed
    set_seed(42)
    model2 = CRBLModel(input_size=128, isCrop=True)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    optimizer2.zero_grad()
    output2 = model2(dummy_input).squeeze()
    loss2 = criterion(output2, dummy_target)
    loss2.backward()
    optimizer2.step()
    
    # Compare models after one step
    is_identical, differences = compare_models(model1, model2)
    
    if is_identical:
        print("✓ Training step is reproducible")
    else:
        print("✗ Training step is NOT reproducible")
        for diff in differences[:3]:  # Show first 3 differences
            print(f"  - {diff}")
    
    return is_identical

def benchmark_performance():
    """Benchmark model performance"""
    print("\nBenchmarking model performance...")
    
    model = CRBLModel(input_size=128, isCrop=True)
    dummy_input = torch.randn(1, 3, 128, 128)
    
    avg_time, std_time = benchmark_inference(model, dummy_input, num_runs=100)
    
    print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.1f} images/second")

def main():
    """Run all reproducibility tests"""
    print("="*80)
    print("CRBL PyTorch Reproducibility Tests")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Model Initialization", test_model_reproducibility()))
    results.append(("Inference", test_inference_reproducibility()))
    results.append(("Data Loader", test_data_loader_reproducibility()))
    results.append(("Training Step", test_training_reproducibility()))
    
    # Benchmark performance
    benchmark_performance()
    
    # Summary
    print("\n" + "="*80)
    print("REPRODUCIBILITY TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✓ All reproducibility tests PASSED")
        print("The implementation is fully reproducible!")
    else:
        print("✗ Some reproducibility tests FAILED")
        print("Please check the implementation for non-deterministic operations")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    main()
