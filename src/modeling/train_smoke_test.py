import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from src.data.build_examples import build_examples, load_training_data

from src.data.label_mapping import create_label_mappings
from src.modeling.tokenize_and_align import tokenize_and_align_example
from src.modeling.prepare_datasets import prepare_split
from src.modeling.build_model import build_model
from src.modeling.tokenize_and_align import MODEL_NAME
from src.data.make_splits import split_examples


TRAIN_SMOKE_SIZE = 200
VAL_SMOKE_SIZE = 50

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
EPOCHS = 1


def evaluate(
    model,
    dataloader,
    device,
    
) -> float:
    """
    Calculate average validation loss.
    """
    
    model.eval()
    
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            
            outputs = model(**batch)
            
            
            
            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1
            
    return total_loss / total_batches 

def main() -> None:
    print("--- CUDA CHECK ---")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Stopping.")
        return
    
    device = torch.device("cuda")
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    torch.cuda.reset_peak_memory_stats(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    df = load_training_data()
    
    
    if df.empty:
        print("No training data loaded. Stopping.")
        return
    
    examples = build_examples(df)

    label2id, id2label = create_label_mappings(examples)

    train_examples, val_examples = split_examples(examples)
    
    # Small subsets only.
    train_examples = train_examples[:TRAIN_SMOKE_SIZE]
    val_examples = val_examples[:VAL_SMOKE_SIZE]
    
    print("\n--- SMOKE TEST DATA ---")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    train_dataset, train_stats = prepare_split(
        train_examples,
        tokenizer,
        label2id,
    )

    val_dataset, val_stats = prepare_split(
        val_examples,
        tokenizer,
        label2id,
    )
    
    
    print(
        f"Training alignment mismatches: "
        f"{train_stats['alignment_mismatches']}"
    )

    print(
        f"Validation alignment mismatches: "
        f"{val_stats['alignment_mismatches']}"
    )
    
    # Dynamically pad each batch only to the
    # longest sequence in that batch.
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
    )
    
    model = build_model(
        label2id,
        id2label,
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    
    print("\n--- TRAINING CONFIG ---")
    print(f"Epochs: {EPOCHS}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Validation batch size: {VAL_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("\n--- STARTING GPU SMOKE TEST ---")
    
    for epoch in range(EPOCHS):
        model.train()
        
        total_train_loss = 0.0
        
        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                key: value.to(
                    device,
                    non_blocking=True,)
                for key, value in batch.items()
            }
            
            # 1. Clear gradients from previous batch.
            optimizer.zero_grad()
            
            # 2. Forward pass.
            outputs = model(**batch)
            
            # 3. Calculate loss.
            loss = outputs.loss
            
            
            # 4. Backpropagation.
            loss.backward()
            
            # 5. Update model weights.
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )
                
        average_train_loss = (
            total_train_loss / len(train_loader)
        )
        
        validation_loss = evaluate(
            model,
            val_loader,
            device,
        )

        print(f"\n--- EPOCH {epoch + 1} COMPLETE ---")

        print(
            f"Average training loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Average validation loss: "
            f"{validation_loss:.4f}"
        )

    print("\n--- GPU MEMORY ---")

    allocated_gb = (
        torch.cuda.memory_allocated(0)
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.memory_reserved(0)
        / 1024**3
    )

    peak_gb = (
        torch.cuda.max_memory_allocated(0)
        / 1024**3
    )

    print(f"Allocated memory: {allocated_gb:.2f} GB")
    print(f"Reserved memory: {reserved_gb:.2f} GB")
    print(f"Peak allocated memory: {peak_gb:.2f} GB")

    print("\nSMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()







    
    
