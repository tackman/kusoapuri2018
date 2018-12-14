async function run(x, modelPath) {
  const model = await tf.loadModel(modelPath);

  let pred = model.predict(x);

  let transposed = tf.transpose(pred);
  const d = await transposed.data();
  return d;
}

// run();
