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
x2.append(epoch+1)

val_loss /= len(val_loader)
val_accuracy = val_correct / len(val_dataset)
f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
validation_f1_history.append(f1_val)

log(classification_report(y_true_val,
                          y_pred_val, target_names=dataset.classes))
log(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}, F1: {f1_val}')

plt.clf()
plt.figure(figsize=(5, 7))
plt.title('Wykres funkcji straty od epoki')
plt.plot(x1, training_loss_history, label='Strata treningu')
plt.plot(x2, validation_loss_history, label='Strata walidacji')
plt.legend()
plt.xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.savefig(f'{experiment_path}/loss.png', bbox_inches="tight")

plt.clf()
plt.figure(figsize=(5, 7))
plt.title('Wykres F1 od epoki')
plt.plot(x1, training_f1_history,
         label='F1 dla danych treningowych')
plt.plot(x2, validation_f1_history,
         label='F1 dla danych walidacyjnych')
plt.legend()
plt.ylim(0, 1)
plt.xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
plt.xlabel('Epoka')
plt.ylabel('F1')
plt.savefig(f'{experiment_path}/f1.png', bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].plot(x1, training_loss_history, label='Strata treningu')
ax[0].plot(x2, validation_loss_history, label='Strata walidacji')
ax[0].legend()
ax[0].set_title('Wykres funkcji straty od epoki')
ax[0].set_xlabel('Epoka')
ax[0].set_ylabel('Strata')
ax[0].set_xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
ax[1].plot(x1, training_f1_history,
           label='F1 dla danych treningowych')
ax[1].plot(x2, validation_f1_history,
           label='F1 dla danych walidacyjnych')
ax[1].legend()
ax[1].set_title('Wykres F1 od epoki')
ax[1].set_xlabel('Epoka')
ax[1].set_ylabel('F1')
ax[1].set_ylim(0, 1)
ax[1].set_xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
plt.savefig(f'{experiment_path}/combined.png', bbox_inches="tight")

cf_matrix = confusion_matrix(y_true_val, y_pred_val)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[
    f'{names[i]} ({i})' for i in dataset.classes], columns=[f'{names[i]} ({i})' for i in dataset.classes])
plt.clf()
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)
plt.title('Macierz pomylek')
plt.savefig(f'{experiment_path}/confusion_matrix.png',
            bbox_inches="tight")

torch.save(model.state_dict(), f'{experiment_path}/model.pth')