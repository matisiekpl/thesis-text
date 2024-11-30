train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)


model = models.efficientnet_b0(weights='DEFAULT')
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device(DEVICE)
model.to(device)

training_loss_history = []
validation_loss_history = []

training_f1_history = []
validation_f1_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0

    if not DRY:
        y_true_train = []
        y_pred_train = []
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            running_loss += loss.item()
        training_loss_history.append(running_loss)
        running_accuracy = running_correct / len(train_dataset)
        f1_train = f1_score(y_true_train, y_pred_train, average='weighted')
        training_f1_history.append(f1_train)

    log(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss/len(train_loader)}')