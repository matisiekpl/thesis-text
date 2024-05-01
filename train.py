train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.efficientnet_b0(weights='DEFAULT')
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device(DEVICE)
model.to(device)

training_loss_history = []
validation_loss_history = []

training_accuracy_history = []
validation_accuracy_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()

        running_loss += loss.item()
    training_loss_history.append(running_loss)
    running_accuracy = running_correct / len(train_dataset)
    training_accuracy_history.append(running_accuracy)
    log(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss/len(train_loader)}')