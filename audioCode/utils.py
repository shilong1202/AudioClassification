import torch
from torch import nn
from tqdm import tqdm


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, device,lr,val_dl,fold):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)

    # Repeat for each epoch
    best_acc = 0
    loss_list = []
    acc_list =[]
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc='Train ')
        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        val_acc = val(model,val_dl,device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'../output/fold/model_fold{fold + 1}_epoch{epoch+1}.pth')

        print(f'fold: {fold},Best Accuracy: {best_acc:.2f}')

        loss_list.append(avg_loss)
        acc_list.append(val_acc)
    return best_acc,loss_list,acc_list




# ----------------------------
# Inference
# ----------------------------
def val(model, val_dl,device):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # print(prediction)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    return acc
