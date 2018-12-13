async function run(x) {
  const model = await tf.loadModel('nets/ep100.tfjs/model.json');

  let pred = model.predict(x);
  //  pred.print();
  console.log(pred.shape);

  let transposed = tf.transpose(pred);
  console.log(transposed.shape);
  const d = await transposed.data();
  return d;
}

// run();
