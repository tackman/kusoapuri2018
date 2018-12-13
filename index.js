async function run(x, modelPath) {

  const model = await tf.loadModel(modelPath);

  let pred = model.predict(x);
  //  pred.print();
  console.log(pred.shape);

  let transposed = tf.transpose(pred);
  console.log(transposed.shape);
  const d = await transposed.data();
  return d;
}

// run();
