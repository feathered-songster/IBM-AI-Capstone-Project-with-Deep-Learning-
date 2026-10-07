Torch workflow notes

- Get access to your data
- Make a DataSet
- Make a DataLoader
- Create a suitable model architercture - by inheriting the nn.Module or using nn.Sequential
- Determine your device and send a model instance  to(device)
- Determine suitable criterion (say nn.CrossEntropyLoss())
- Create an optimizer
- Decide on the number of epochs and start the main training loop
    * set up the loop over the loader's batches on the relevant device
	* set the model to train state
	* zero_grad() the optimizer
	* calculate outputs of the model over the batch
	* calculate loss by using criterion
	* loss.backward()
	* optimizer.step()
	* calculate any metrics to keep track of during training
    * set model to .eval()
    * with torch.no_grad(): set up a loop over the validation batches on the relevant device
    * calculate and display relevant epoch end validation metrics
    
