
import * as ort from 'onnxruntime-web';

async function runPrediction(inputArray) {
    const session = await ort.InferenceSession.create('best_model.onnx');
    const tensor = new ort.Tensor('float32', inputArray, [1, 8]);
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const output = results.output.data;
    return output;
}

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const inputArray = [
        parseFloat(formData.get('pregnancies')),
        parseFloat(formData.get('glucose')),
        parseFloat(formData.get('bloodpressure')),
        parseFloat(formData.get('skinthickness')),
        parseFloat(formData.get('insulin')),
        parseFloat(formData.get('bmi')),
        parseFloat(formData.get('dpf')),
        parseFloat(formData.get('age'))
    ];
    const output = await runPrediction(inputArray);
    document.getElementById('result').innerText = `Predicted Class: ${output[0]}`;
});
