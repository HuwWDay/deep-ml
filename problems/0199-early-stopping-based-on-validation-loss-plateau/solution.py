def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
    waiting = 0 
    best_loss = val_losses[0]
    out = [False]
	for loss in val_losses[1:]:
        if loss < best_loss - min_delta:
            best_loss = loss
        else:
            waiting += 1
        
        
        if waiting >= patience:
            out.append(True)
            waiting = 0
        else:
            out.append(False)
    return out
