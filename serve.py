@app.route('/predict/<revision>', methods=['POST'])
def predict(revision):
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return 'Invalid file'
    image_stream = BytesIO(file.read())
    image = transform(Image.open(image_stream).convert('RGB'))

    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        result = {}
        for i, p in enumerate(outputs[0]):
            percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
            print(f'{names[classes[i]]}: {percent.item():.4f}%')
            result[names[classes[i]]] = percent.item()
        return result