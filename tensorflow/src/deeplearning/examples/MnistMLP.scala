package deeplearning.examples

import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.{tf, _}
import org.slf4j.LoggerFactory


// Handwritten Digit Classification on MNIST Dataset using simple multilayer perceptron in Tensorflow

object MnistMLP {
  private[this] val logger = Logger(LoggerFactory.getLogger(MnistMLP.getClass))

  def main(args: Array[String]): Unit = {


    //Defining Hyperparameters of the Neural Network
    val numHidden    = 512
    val numOutputs   = 10
    val learningRate = 0.01
    val batchSize    = 128
    val numEpochs    = 10

    // downloading and loading the MNIST images
    val dataSet = MNISTLoader.load(Paths.get("tensorflow/datasets/MNIST"))

    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
    val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
    val testImages  = tf.data.TensorSlicesDataset(dataSet.testImages)
    val testLabels  = tf.data.TensorSlicesDataset(dataSet.testLabels)

    val trainData =
      trainImages.zip(trainLabels)
          .repeat()
          .shuffle(10000)
          .batch(batchSize)
          .prefetch(10)

    val evalTrainData = trainImages.zip(trainLabels).batch(1000).prefetch(10)
    val evalTestData = testImages.zip(testLabels).batch(1000).prefetch(10)

    // neural network architecture
    val input = tf.learn.Input(UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2)))
    val trainInput = tf.learn.Input(UINT8, Shape(-1))

    //Flatten Image into a Single Vector and defining the different layers
    val layer = tf.learn.Flatten("Input/Flatten") >>
      tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Linear("Layer_1/Linear", numHidden, weightsInitializer = GlorotUniformInitializer()) >>
      tf.learn.ReLU("Layer_1/ReLU") >>
      tf.learn.Linear("OutputLayer/Linear", numOutputs, weightsInitializer = GlorotUniformInitializer())

    val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)

    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
        tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.GradientDescent(learningRate)

    val model = tf.learn.Model.supervised(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    val summariesDir = Paths.get("tensorflow/temp/mnist-mlp")
    val accMetric = tf.metrics.MapMetric(
      (v: (Output, Output)) => (v._1.argmax(-1), v._2), tf.metrics.Accuracy())
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some((60000/batchSize)*numEpochs)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.Evaluator(
          log = true, datasets = Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
          metrics = Seq(accMetric), trigger = tf.learn.StepHookTrigger(1000), name = "Evaluator"),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))

    // train the model
    estimator.train(() => trainData)

    def accuracy(images: Tensor, labels: Tensor): Float = {
      val predictions = estimator.infer(() => images)
      predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
    }

    // evaluate model performance
    logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}
