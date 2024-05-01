model.eval()
val_loss = 0.0
val_correct = 0
y_true_val = []
y_pred_val = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()
        y_true_val.extend(labels.cpu().numpy())
        y_pred_val.extend(predicted.cpu().numpy())

validation_loss_history.append(val_loss)

val_loss /= len(val_loader)
val_accuracy = val_correct / len(val_dataset)
validation_accuracy_history.append(val_accuracy)
f1_val = f1_score(y_true_val, y_pred_val, average='weighted')

log(classification_report(y_true_val,  y_pred_val, target_names=dataset.classes))
log(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}, F1: {f1_val}')

cf_matrix = confusion_matrix(y_true_val, y_pred_val)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[
    f'{names[i]} ({i})' for i in dataset.classes], columns=[f'{names[i]} ({i})' for i in dataset.classes])
plt.cla()
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)
plt.title('Macierz pomyłek')
plt.savefig(f'{experiment_path}/confusion_matrix.png', bbox_inches="tight")

plt.cla()
plt.title('Wykres funkcji straty od epoki')
plt.plot(training_loss_history, label='Strata treningu')
plt.plot(validation_loss_history, label='Strata walidacji')
plt.legend()
plt.savefig(f'{experiment_path}/loss.png', bbox_inches="tight")

plt.cla()
plt.title('Wykres dokładności od epoki')
plt.plot(training_accuracy_history, label='Dokładność dla danych treningowych')
plt.plot(validation_accuracy_history, label='Dokładność dla danych walidacyjnych')
plt.legend()
plt.ylim(0, 1)
plt.savefig(f'{experiment_path}/acc.png', bbox_inches="tight")

torch.save(model.state_dict(), f'{experiment_path}/model.pth')